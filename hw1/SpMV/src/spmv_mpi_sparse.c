/* HW1 Task 2: distributed CSR SpMV, exchanging only the x entries each rank
 * needs.
 *
 * ONE OF THE TWO FILES YOU SUBMIT. Its partner is spmv_mpi_allgather.c
 * (Task 1). Same four functions, same rules -- the only difference is the
 * communication. Each file is built and graded on its own, so this one must be
 * complete by itself: do not #include the other, and do not assume anything it
 * allocated exists here.
 *
 * The allgather baseline moves the WHOLE x to EVERY rank, most of which a
 * given rank never touches. Here you send each rank only the entries its local
 * rows actually reference. The plan must be built ONCE in spmv_mpi_setup();
 * the harness times setup separately, so a plan rebuilt inside the timed
 * spmv_mpi() shows up as inflated per-iteration cost.
 *
 * Rules:
 *   - no main(), no MPI_Init, no MPI_Finalize, no reading the matrix file
 *   - use the `comm` you are given, not MPI_COMM_WORLD
 *
 * Build and test:
 *   make spmv_mpi_sparse
 *   mpirun -n 4 ./spmv_mpi_sparse matrices/dictionary28.mtx
 * The harness prints PASS or FAIL against the sequential CSR baseline.
 */

#include <stdlib.h>
#include <string.h>
#include "spmv_mpi.h"

/* A sketch of what a plan needs. Two halves, and you have to build both:
 * what you RECEIVE (which remote columns your rows touch, and from whom) and
 * what you SEND (which of your x entries other ranks asked for). Neither side
 * can be guessed locally -- one exchange in setup tells each rank what its
 * peers want. Rename, drop or add fields as your design requires. */
typedef struct spmv_ctx
{
    int rank, nranks;
    int my_start, my_count;

    /* receive side */
    int * recv_count;      // how many x entries from each rank
    int * recv_disp;       // offsets for unpacking received buffer
    int * recv_idx;        // global indices to scatter into x_buf
    float * recv_buf;      // packed incoming elements

    /* send side */
    int * send_count;      // how many x entries requested by each rank
    int * send_disp;       // offsets for packing send buffer
    int * send_idx;        // which LOCAL x entries to pack
    float * send_buf;      // packed outgoing elements

    /* Where the received entries live during spmv(). 
     * Strategy chosen: Keep a full-length x_buf and scatter into it.
     * Why: It uses O(num_cols) memory but keeps the compute loop exactly 
     * identical to the baseline, minimizing complexity and branches. */
    float * x_buf;
    MPI_Request * reqs;    // Handles for non-blocking communication

    comm_stats stats;      // filled in for Task 3d
} spmv_ctx;


int spmv_mpi_setup(const csr_local * A, const vec_partition * xpart,
                   MPI_Comm comm, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    MPI_Comm_rank(comm, &c->rank);
    MPI_Comm_size(comm, &c->nranks);
    c->my_start = xpart->displs[c->rank];
    c->my_count = xpart->counts[c->rank];

    /* 3. Allocate buffers. 
     * nranks is strictly >= 1, so calloc is perfectly safe and necessary here 
     * to prevent garbage memory values from crashing the MPI counts. */
    c->recv_count = (int*)calloc(c->nranks, sizeof(int));
    c->send_count = (int*)calloc(c->nranks, sizeof(int));
    c->recv_disp  = (int*)calloc(c->nranks + 1, sizeof(int));
    c->send_disp  = (int*)calloc(c->nranks + 1, sizeof(int));
    
    // alloc_array for the lookup buffer in case the matrix has no columns
    c->x_buf = (float*)alloc_array(A->num_cols, sizeof(float));

    /* 1. Scan A->col_idx. For each distinct global column j your rows touch,
     *    find who owns it; skip the ones you own. */
    char *needed = (char*)calloc((A->num_cols > 0 ? A->num_cols : 1), 1);
    
    // A->row_ptr[A->num_rows] gives the total number of non-zeros (nnz)
    for (int i = 0; i < A->row_ptr[A->num_rows]; i++) {
        int col = A->col_idx[i];
        if (col < c->my_start || col >= c->my_start + c->my_count)
            needed[col] = 1;
    }

    // Count how many we need from each owner
    for (int col = 0; col < A->num_cols; col++) {
        if (needed[col]) {
            int r = 0;
            while (r < c->nranks - 1 && col >= xpart->displs[r] + xpart->counts[r]) r++;
            c->recv_count[r]++;
        }
    }

    /* 2. Tell each owner what you want, and learn what others want from you. */
    MPI_Alltoall(c->recv_count, 1, MPI_INT, c->send_count, 1, MPI_INT, comm);

    for (int i = 0; i < c->nranks; i++) {
        c->recv_disp[i+1] = c->recv_disp[i] + c->recv_count[i];
        c->send_disp[i+1] = c->send_disp[i] + c->send_count[i];
        if (c->send_count[i] > 0) c->stats.messages++;
    }

    int total_recv = c->recv_disp[c->nranks];
    int total_send = c->send_disp[c->nranks];
    
    // These specific arrays CAN legitmately be size 0 on isolated ranks, so we must use alloc_array
    c->recv_idx = (int*)alloc_array(total_recv, sizeof(int));
    c->send_idx = (int*)alloc_array(total_send, sizeof(int));

    // Pack the exact global column indices we need to request
    int *temp = (int*)calloc(c->nranks, sizeof(int));
    for (int col = 0; col < A->num_cols; col++) {
        if (needed[col]) {
            int r = 0;
            while (r < c->nranks - 1 && col >= xpart->displs[r] + xpart->counts[r]) r++;
            c->recv_idx[c->recv_disp[r] + temp[r]++] = col;
        }
    }
    free(temp); free(needed);

    // Exchange the exact requested column indices
    MPI_Alltoallv(c->recv_idx, c->recv_count, c->recv_disp, MPI_INT,
                  c->send_idx, c->send_count, c->send_disp, MPI_INT, comm);

    // Translate requested global indices into local x offsets for quick packing in spmv_mpi()
    for (int i = 0; i < total_send; i++) {
        c->send_idx[i] -= c->my_start;
    }

    c->recv_buf = (float*)alloc_array(total_recv, sizeof(float));
    c->send_buf = (float*)alloc_array(total_send, sizeof(float));
    c->reqs = (MPI_Request*)alloc_array(c->nranks * 2, sizeof(MPI_Request));

    /* Counters: messages and payload bytes for ONE SpMV. */
    c->stats.bytes_sent = (long long)total_send * sizeof(float);

    *ctx = c;
    return 0;
}


void spmv_mpi(void * ctx, const csr_local * A, const float * x_local,
              float * y_local, MPI_Comm comm)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    int req_idx = 0;

    /* STEP 1 -- move only the planned entries. Pack from x_local using
     * c->send_idx, post the receives and sends, wait. */
     
    for (int i = 0; i < c->send_disp[c->nranks]; i++) {
        c->send_buf[i] = x_local[c->send_idx[i]];
    }

    // Embed our own local slice directly into the lookup buffer
    if (c->my_count > 0) {
        memcpy(c->x_buf + c->my_start, x_local, c->my_count * sizeof(float));
    }

    for (int i = 0; i < c->nranks; i++) {
        if (c->recv_count[i] > 0) {
            MPI_Irecv(c->recv_buf + c->recv_disp[i], c->recv_count[i], MPI_FLOAT, 
                      i, 0, comm, &c->reqs[req_idx++]);
        }
        if (c->send_count[i] > 0) {
            MPI_Isend(c->send_buf + c->send_disp[i], c->send_count[i], MPI_FLOAT, 
                      i, 0, comm, &c->reqs[req_idx++]);
        }
    }

    if (req_idx > 0) {
        MPI_Waitall(req_idx, c->reqs, MPI_STATUSES_IGNORE);
    }

    // Unpack received entries into their proper global indices in x_buf
    for (int i = 0; i < c->recv_disp[c->nranks]; i++) {
        c->x_buf[c->recv_idx[i]] = c->recv_buf[i];
    }

    /* STEP 2 -- local computation, exactly as in the allgather version except
     * for where a column's x value is read from. */
    for (int i = 0; i < A->num_rows; i++) {
        float sum = 0.0f;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
            sum += A->vals[k] * c->x_buf[A->col_idx[k]];
        }
        y_local[i] = sum;
    }
}


void spmv_mpi_stats(void * ctx, comm_stats * out)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    *out = c->stats;
}


void spmv_mpi_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    free(c->recv_count); free(c->recv_disp); free(c->recv_idx); free(c->recv_buf);
    free(c->send_count); free(c->send_disp); free(c->send_idx); free(c->send_buf);
    free(c->reqs); free(c->x_buf);
    free(c);
}
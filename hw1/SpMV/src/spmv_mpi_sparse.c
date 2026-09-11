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

    /* receive side */
    int   n_recv_peers;    //how many ranks you receive from
    int * recv_peer;       //their ranks
    int * recv_count;      //how many x entries from each

    /* send side */
    int   n_send_peers;
    int * send_peer;
    int * send_count;
    int * send_idx;        //which LOCAL x entries to pack, peer by peer

    /* Where the received entries live during spmv(). Some implementations
     * keep a full-length x_full and scatter into it (simple, but O(num_cols)
     * memory per rank); others keep a compact buffer and remap col_idx once in
     * setup so the kernel indexes it directly (less memory, and the kernel
     * gets better locality). Either is acceptable -- say which you chose and
     * why in the report. */
    float * x_buf;

    comm_stats stats;      //filled in for Task 3d
} spmv_ctx;


int spmv_mpi_setup(const csr_local * A, const vec_partition * xpart,
                   MPI_Comm comm, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    MPI_Comm_rank(comm, &c->rank);
    MPI_Comm_size(comm, &c->nranks);

    /* TODO -- build the plan, once:
     *
     *  1. Scan A->col_idx. For each distinct global column j your rows touch,
     *     partition_owner(xpart, j) says who owns it; skip the ones you own.
     *     That gives your receive side. Sorting the needed columns per owner
     *     keeps both ends agreeing on the order without sending it twice.
     *  2. Tell each owner what you want, and learn what others want from you.
     *     Every rank has an identical xpart, so you can compute step 1 with no
     *     communication -- but only the requester knows which entries it
     *     needs, so the send side does require an exchange here in setup.
     *  3. Allocate the buffers spmv_mpi() will reuse every iteration. Use
     *     alloc_array (include/alloc.h) for anything that can legitimately be
     *     empty: a rank whose rows are entirely local sends and receives
     *     nothing, and malloc(0) may return NULL.
     *
     * Nothing below this point should allocate or plan.
     */

    /* TODO: your Task 3d counters -- messages and payload bytes for ONE SpMV.
     * Here they are just your own MPI_Isend/MPI_Send calls and their payloads;
     * no cost model to argue about, unlike the allgather. */
    c->stats.messages   = 0;
    c->stats.bytes_sent = 0;

    (void)A;
    (void)xpart;

    *ctx = c;
    return 0;
}


void spmv_mpi(void * ctx, const csr_local * A, const float * x_local,
              float * y_local, MPI_Comm comm)
{
    spmv_ctx * c = (spmv_ctx*)ctx;

    /* STEP 1 -- move only the planned entries. Pack from x_local using
     * c->send_idx, post the receives and sends, wait. No allocation, no
     * planning, no collective over all ranks.
     *
     * TODO */
    (void)x_local;
    (void)comm;
    (void)c;

    /* STEP 2 -- local computation, exactly as in the allgather version except
     * for where a column's x value is read from.
     *
     * TODO */
    for (int i = 0; i < A->num_rows; i++)
        y_local[i] = 0.0f;
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
    free(c->recv_peer);  free(c->recv_count);
    free(c->send_peer);  free(c->send_count);  free(c->send_idx);
    free(c->x_buf);
    free(c);
}

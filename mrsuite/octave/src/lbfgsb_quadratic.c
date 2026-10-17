/*==================================================================================================
 * lbfgsb_quadratic.c
 *
 * Authors: J. P. Merkofer (j.p.merkofer@tue.nl)
 *
 * Created: 2026-09-27
 *
 * Purpose: Osprey's L-BFGS-B-C driver loop for the one kind of problem Osprey gives it,
 *          with f and g computed here instead of by a callback into the interpreter:
 *
 *              f(x) = norm(A + M*x - b)^2        (A may be empty)
 *              g(x) = 2*(H*x - c)
 *
 *          The loop is lbfgsb_wrapper.c's (Stephen Becker, BSD licence), so the iterates are
 *          the same; the arithmetic is Octave's, operation for operation - the same BLAS
 *          dgemv, element-wise steps in Octave's order, and Octave's scaled 2-norm - so f and
 *          g are bit for bit what the interpreter would have returned. The one part that
 *          depends on how Octave itself was compiled is whether that norm's accumulator was
 *          fused into multiply-adds (gcc does, clang here does not); both are built, and the
 *          caller picks the one its Octave agrees with (see shims/lbfgsb.m).
 *
 * Usage:   [f, x, task, iterations, total] = lbfgsb_quadratic(m, x, l, u, nbd, M, A, b, H, c,
 *                                              factr, pgtol, iprint, maxIts, maxTotalIts, fused)
 *          nrm = lbfgsb_quadratic('norm', v, fused)      Octave's norm(v), for calibration
 *
 * The driver loop is lbfgsb_wrapper.c's, from L-BFGS-B-C (Osprey's libraries/L-BFGS-B-C),
 * under its licence:
 *
 *   BSD 3-clause license
 *   Copyright (c) 2015, Stephen Becker
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *================================================================================================*/
#include <math.h>
#include <stddef.h>
#include "lbfgsb.h"

/* the state arrays setulb keeps between calls, sized as in lbfgsb_wrapper.c */
#define LENGTH_LSAVE 4
#define LENGTH_ISAVE 44
#define LENGTH_DSAVE 29

extern void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
                   const double *a, const int *lda, const double *x, const int *incx,
                   const double *beta, double *y, const int *incy, size_t trans_len);

/*------------------------------------------------------------------------------------------------
 * Octave's vector 2-norm (liboctave/numeric/oct-norm.cc, norm_accumulator_2), with its two
 * accumulation steps either as written or fused the way gcc contracts them. volatile keeps
 * this compiler from fusing the plain version behind our back.
 *----------------------------------------------------------------------------------------------*/
static double octave_norm(const double *v, int n, int fused) {
    double scl = 0, sum = 1;
    for (int i = 0; i < n; i++) {
        double t = fabs(v[i]);
        if (scl == t)
            sum += 1;
        else if (scl < t) {
            double q = scl / t;
            volatile double p = q * q;
            if (fused)
                sum = fma(sum, p, 1.0);
            else {
                volatile double s = sum * p;
                sum = s + 1;
            }
            scl = t;
        }
        else if (t != 0) {
            double q = t / scl;
            if (fused)
                sum = fma(q, q, sum);
            else {
                volatile double p = q * q;
                sum += p;
            }
        }
    }
    return scl * sqrt(sum);
}

/*------------------------------------------------------------------------------------------------
 * f and g at x, as Octave evaluates norm(A + M*x - b)^2 and 2*(H*x - c)
 *----------------------------------------------------------------------------------------------*/
static double evaluate(const double *x, int r, int n, const double *M, const double *A,
                       const double *b, const double *H, const double *c, double *y,
                       double *g, int fused) {
    const double one = 1.0, zero = 0.0;
    const int inc = 1;
    dgemv_("N", &r, &n, &one, M, &r, x, &inc, &zero, y, &inc, 1);
    for (int i = 0; i < r; i++) {
        if (A) y[i] = A[i] + y[i];
        y[i] = y[i] - b[i];
    }
    dgemv_("N", &n, &n, &one, H, &n, x, &inc, &zero, g, &inc, 1);
    for (int i = 0; i < n; i++) g[i] = 2 * (g[i] - c[i]);
    /* Octave's ^ is std::pow from the system library, which need not equal the exact
       square (Apple's is a bit off on some values); a literal 2.0 would let the compiler
       turn pow into x*x, so the exponent goes through a volatile */
    volatile double two = 2.0;
    return pow(octave_norm(y, r, fused), two);
}

/*------------------------------------------------------------------------------------------------
 * gateway
 *----------------------------------------------------------------------------------------------*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs == 3 && mxIsChar(prhs[0])) {
        plhs[0] = mxCreateDoubleScalar(octave_norm(mxGetPr(prhs[1]),
            (int)mxGetNumberOfElements(prhs[1]), (int)mxGetScalar(prhs[2])));
        return;
    }
    if (nrhs != 16) mexErrMsgTxt("lbfgsb_quadratic: expected 16 inputs");

    integer m = (integer)mxGetScalar(prhs[0]);
    integer n = (integer)mxGetM(prhs[1]);
    int r = (int)mxGetM(prhs[5]), nn = (int)n;
    /* the caller trusts the shapes Osprey builds; a mismatch is a bug, so say so */
    for (int k = 5; k <= 9; k++)
        if (!mxIsDouble(prhs[k]) || mxIsComplex(prhs[k]) || mxIsSparse(prhs[k]))
            mexErrMsgTxt("lbfgsb_quadratic: M, A, b, H and c must be real, full doubles");
    if ((int)mxGetN(prhs[5]) != nn || (int)mxGetM(prhs[8]) != nn || (int)mxGetN(prhs[8]) != nn
            || (int)mxGetNumberOfElements(prhs[7]) != r || (int)mxGetNumberOfElements(prhs[9]) != nn
            || (mxGetNumberOfElements(prhs[6]) && (int)mxGetNumberOfElements(prhs[6]) != r))
        mexErrMsgTxt("lbfgsb_quadratic: expected M r x n, A empty or r x 1, b r x 1, "
                     "H n x n and c n x 1");

    plhs[1] = mxDuplicateArray(prhs[1]);
    double *x = mxGetPr(plhs[1]), *l = mxGetPr(prhs[2]), *u = mxGetPr(prhs[3]);
    const double *M = mxGetPr(prhs[5]), *b = mxGetPr(prhs[7]), *H = mxGetPr(prhs[8]),
                 *c = mxGetPr(prhs[9]);
    const double *A = mxGetNumberOfElements(prhs[6]) ? mxGetPr(prhs[6]) : NULL;
    double factr = mxGetScalar(prhs[10]), pgtol = mxGetScalar(prhs[11]);
    integer iprint = (integer)mxGetScalar(prhs[12]);
    int iterMax = (int)mxGetScalar(prhs[13]), total_iterMax = (int)mxGetScalar(prhs[14]);
    int fused = (int)mxGetScalar(prhs[15]);

    /* nbd as lbfgsb.m passes it: int64 off Windows, int32 on it */
    integer *nbd = (integer *)mxMalloc(n * sizeof(integer));
    if (mxIsInt64(prhs[4])) {
        const long long *src = (const long long *)mxGetData(prhs[4]);
        for (int i = 0; i < nn; i++) nbd[i] = (integer)src[i];
    } else if (mxIsInt32(prhs[4])) {
        const int *src = (const int *)mxGetData(prhs[4]);
        for (int i = 0; i < nn; i++) nbd[i] = (integer)src[i];
    } else {
        const double *src = mxGetPr(prhs[4]);
        for (int i = 0; i < nn; i++) nbd[i] = (integer)src[i];
    }

    double f = 0;
    double *g = (double *)mxMalloc(n * sizeof(double));
    double *y = (double *)mxMalloc(r * sizeof(double));
    double *wa = (double *)mxMalloc((2 * m * n + 5 * n + 11 * m * m + 8 * m) * sizeof(double));
    integer *iwa = (integer *)mxMalloc(3 * n * sizeof(integer));
    integer task = (integer)START, csave = (integer)1;
    logical lsave[LENGTH_LSAVE];
    integer isave[LENGTH_ISAVE];
    double dsave[LENGTH_DSAVE];
    integer iterations = 0, total_iterations = 0;

    /* lbfgsb_wrapper.c's loop, the callback replaced by evaluate() */
    while ((iterations < iterMax) && (total_iterations < total_iterMax)) {
        total_iterations++;
        setulb(&n, &m, x, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, &task, &iprint,
               &csave, lsave, isave, dsave);
        if (IS_FG(task)) {
            f = evaluate(x, r, nn, M, A, b, H, c, y, g, fused);
            continue;
        }
        if (task == NEW_X) {
            iterations++;
            continue;
        }
        break;
    }

    plhs[0] = mxCreateDoubleScalar(f);
    if (nlhs >= 3) plhs[2] = mxCreateDoubleScalar((double)task);
    if (nlhs >= 4) plhs[3] = mxCreateDoubleScalar((double)iterations);
    if (nlhs >= 5) plhs[4] = mxCreateDoubleScalar((double)total_iterations);

    mxFree(nbd); mxFree(g); mxFree(y); mxFree(wa); mxFree(iwa);
}

#' m-fold Cross-validation for Generalised Linear Models and specific divergence
#'
#' \code{CVmFold} returns the m-fold Cross-validation prediction error of different divergences
#' for generalised linear models.
#'
#' This function computes the m-fold Cross-validation (CV) of a Generalised Linear Models \code{family}
#' to assesses the prediction error according to a specific \code{divergence}. It is called inside
#' \code{\link[panning]{InitialStep}} and \code{\link[panning]{GeneralStep}} functions, the two
#' main functions of the Panning Algorithm.
#'
#' In the case \code{divergence = "classification"}, it is possible to have asymmetric
#' classification errors by setting the \code{W} matrix (rows: estimated y; columns: true y) (see the
#' example below). For logistic regression (runned with \code{\link[stats]{glm}}), the cutoff
#' value \code{C0} determines whether the prediction takes value 0 (prediction <=\code{C0}) or 1
#' (prediction >\code{C0}). For multinomial regression, \code{increasing=TRUE} states
#' \code{y}>=1 with unit increments (it makes \code{\link[panning]{CVmFold}} runs faster).
#'
#' Attention should be taken on how the estimated values of \code{y} should be returned,
#' and choose \code{type} accordingly. See the example below on logistic regression.
#'
#' @param y             is a (n x 1) vector of response variable.
#' @param X             is a (n x p) matrice of predictors.
#' @param m             is the number of folds.
#' @param K             is the number of repetitions.
#' @param family        the family object for \code{\link[stats]{glm}} or \code{family = "multinomial"}
#'      to use \code{\link[nnet]{multinom}}.
#' @param type          the type of prediction required for \code{\link[stats]{predict}} function.
#' @param divergence    the type of divergence. \code{divergence = "L1"} is the L1-norm error.
#'      \code{divergence = "sq.error"} is the squared error. \code{divergence = "classification"} gives the classification error.
#' @param C0            is a cutoff value between (0,1)
#' @param W             is a matrix of weights for classification errors (if \code{divergence = "classification"}).
#'      If \code{W=NULL} (default), \code{W} has 0 elements in the diagonal (good predictions) and 1s elsewhere.
#' @param increasing    is a boolean characterising \code{y} (see details).
#' @param trace         if \code{trace = TRUE}, hide the warnings of the fitting method.
#' @param ...           additional arguments affecting the fitting method (see \code{\link[stats]{glm}}
#'      or \code{\link[nnet]{multinom}}).
#'
#' @return \code{CVmFold} returns a single numeric value assessing the estimated prediction error.
#'
#' @author Samuel Orso \email{Samuel.Orso@unige.ch}
#'
#' @seealso \code{\link[stats]{glm}}, \code{\link[stats]{family}}, \code{\link[stats]{predict.glm}},
#' \code{\link[panning]{InitialStep}}, \code{\link[panning]{GeneralStep}}
#'
#' @examples
#' \dontrun{
#' ### Binary data
#' # load the data
#' library(MASS)
#' data("birthwt")
#' y <- birthwt$low
#' X <- as.matrix(birthwt)[,-1]
#'
#' ## logistic regression with glm()
#' # L1 error
#' set.seed(123)
#' CVmFold(y = y, X = X, family = binomial(link = 'logit'), divergence = "L1",
#'      type = "response", trace = FALSE, control = list(maxit=100) )
#'
#' # Squared error
#' set.seed(123)
#' CVmFold(y = y, X = X, family = binomial(link = 'logit'), divergence = "sq.error",
#'      type = "response", trace = FALSE, control = list(maxit=100) )
#'
#' # misclassification error
#' set.seed(123)
#' CVmFold(y = y, X = X, family = binomial(link = 'logit'), divergence = "classification",
#'      type = "response", trace = FALSE, control = list(maxit=100) )
#'
#' # asymmetric misclassification error
#' Weight <- matrix(c(0,1.5,0.5,0),2,2)
#' set.seed(123)
#' CVmFold(y = y, X = X, family = binomial(link = 'logit'), divergence = "classification",
#'      W = Weight, type = "response", trace = FALSE, control = list(maxit=100) )
#'
#' ## logistic regression with multinom()
#' # L1 error
#' set.seed(123)
#' CVmFold(y = y, X = X, family = "multinomial", divergence = "L1", type = "probs" )
#'
#' # Squared Error
#' set.seed(123)
#' CVmFold(y = y, X = X, family = "multinomial", divergence = "sq.error", type = "probs" )
#'
#' # misclassification error
#' y <- y+1L
#' set.seed(123)
#' CVmFold(y = y, X = X, family = "multinomial", divergence = "classification",
#'      type = "class", increasing = TRUE )
#'
#' # asymmetric misclassification error
#' set.seed(123)
#' CVmFold(y = y, X = X, family = "multinomial", divergence = "classification",
#'      type = "class", W = Weight, increasing = TRUE )
#'
#' ### Count data
#' counts <- c(18,17,15,20,10,20,25,13,12)
#' outcome <- gl(3,1,9)
#' treatment <- gl(3,3)
#' set.seed(123)
#' CVmFold(y = counts, X = cbind(outcome, treatment), m = 3, K = 30, family = poisson(),
#'      divergence = "L1" )
#' }
#' @importFrom nnet multinom
#' @importFrom stats glm
#' @import MASS
#' @export
CVmFold <- function(y, X, m = 10L, K = 10L, family, type = NULL, divergence, C0 = 0.5, W = NULL,
                    increasing = FALSE, trace = TRUE, ... ){
        # Initialisation:
        n <- length(y)          # number of observations
        nc <- ceiling(n/m)      # number of columns for m-fold-CV, m is number of rows
        ne <- nc*m - n          # number of extra observations needed for full matrix in m-fold-CV
        pred.error <- matrix(nrow=m,ncol=K) # matrix of prediction errors
        imX <- is.matrix(X)
        if( divergence == "classification" ) kl <- nlevels(as.factor(y)) # number of classes

        # m-fold-cross-validation:
        for ( j in seq_len(K) ){
                # Construction of a matrix of splits for m-fold-CV:
                if( ne == 0) rs <- matrix(sample.int( n, replace = FALSE ), nrow=m, ncol=nc) else
                        rs <- matrix(c(sample.int( n, replace = FALSE ), rep(NA,ne)), nrow=m, ncol=nc)

                for ( i in seq_len(m) ) {
                        # Seperating training and test datasets
                        i.test <- na.omit(rs[i,])
                        i.train <- na.omit(c(rs[-i,]))

                        y.cv.train <- y[i.train]
                        y.cv.test <- y[i.test]
                        if(imX) X.cv.train <- X[i.train,]       else X.cv.train <- X[i.train]
                        if(imX) X.cv.test <- X[i.test,]         else X.cv.test <- X[i.test]

                        # Fitting the model
                        if( !trace ) options(warn = -1)
                        if( !is.list(family) ){
                                fit <- multinom(y ~ ., data = data.frame(y=y.cv.train, var=X.cv.train),
                                                trace = FALSE, ... )
                        } else {
                                fit <- glm(y ~ ., data = data.frame(y=y.cv.train, var=X.cv.train),
                                                                   family = family, ...)
                        }
                        if( !trace ) options(warn = 0)

                        # Predicting the model
                        y.hat <- predict(fit, newdata = data.frame(var=X.cv.test), type = type)

                        # Assessing the prediction errors
                        if( divergence == "classification" )
                        {
                                # Weight of misclassification
                                if( is.null(W) ) W <- matrix(1,kl,kl) - diag(kl)

                                # Logistic with glm() case
                                if( is.list(family) && family$family == "binomial" )
                                {
                                        y.hat <- ceiling(y.hat - C0) + 1L
                                        y.cv.test <- y.cv.test + 1L
                                        pred.error[i,j] <- sum(W[cbind(y.hat,y.cv.test)])/length(i.test)
                                }else{
                                        # Multiclasses with multinom() case
                                        if( increasing )
                                        {
                                                pred.error[i,j] <- sum(W[cbind(y.hat,y.cv.test)])
                                        }else{
                                                y.all <- c(y.hat, y.cv.test)
                                                nf <- length(unique(y.all))
                                                y.all <- as.integer(factor(y.all, labels = seq_len(nf)))
                                                pred.error[i,j] <- sum(W[matrix(y.all,ncol=2)])/length(i.test)
                                        }
                                }
                        }

                        if( divergence == "L1" )
                        {
                                pred.error[i,j] <- sum( abs(y.hat - y.cv.test) )/length(i.test)
                        }

                        if( divergence == "sq.error" )
                        {
                                pred.error[i,j] <- sum( (y.hat - y.cv.test)^2 )/length(i.test)
                        }
                }
        }
        return(sum(pred.error)/(K*m))
}

# function to combine results of foreach loops
# From : http://stackoverflow.com/questions/19791609/saving-multiple-outputs-of-foreach-dopar-loop
comb <- function(x, ...) {
        lapply(seq_along(x),
               function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
}


#' Initial step of the panning algorithm
#'
#' \code{InitialStep} computes the intial step of the Panning Algorithm.
#'
#' This function computes exhaustively the m-fold Cross-validation (CV) prediction error
#' for all the \eqn{\left( {\begin{array}{*{20}{c}} p \\ d \end{array}} \right)}{C(p,d)} possible models of size \code{d} by calling
#' the \code{\link[panning]{CVmFold}} function. If \code{B=NULL} (default), then
#' \code{B} is set to be equal to \eqn{\left( {\begin{array}{*{20}{c}} p \\ d \end{array}} \right)}{C(p,d)}.
#'
#' If \code{B} takes a positive integer value smaller than the total number of models \eqn{\left( {\begin{array}{*{20}{c}} p \\ d \end{array}} \right)}{C(p,d)},
#' then the function computes the CV prediction errors for \code{B} models of size \code{d} randomly selected.
#' In this case, it is possible to set the \code{seed} for reproducibility.
#'
#' At this stage, the algorithm does not allow for interaction terms among variables.
#'
#' This function is computationnaly time consuming proportionally to the size of \code{B}.
#'
#' @param y,X,m,K,family,type,divergence,C0,W,increasing,trace,...   (see function \code{\link[panning]{CVmFold}})
#' @param d             the dimension of the model of interest (intercept is always included).
#' @param alpha         the level of the quantile of the prediction errors.
#' @param B             the number of bootstrap replicates.
#' @param seed          the seed for the random number generator.
#' @param proc          number of processor(s) for parallelisation.
#'
#' @return \code{InitialStep} returns a list with the following components:
#' \describe{
#'      \item{\code{Ids}}{is the set \eqn{I_d^*} of indices of predictors with prediction errors
#'      \code{cv.error}<= \code{q.alpha}.}
#'      \item{\code{Sds}}{is the set \eqn{S_d^*} of models of size \code{d} with
#'      prediction errors \code{cv.error}<= \code{q.alpha}.}
#'      \item{\code{cv.error}}{is a (\code{B} x 1) vector of CV predictions errors.}
#'      \item{\code{q.alpha}}{is the empirical \code{alpha}-quantile computed on \code{cv.error}.}
#'      \item{\code{var.mat}}{is a (\code{B}x\code{d}) matrix of indices of the explored models.}
#' }
#' The indices returned by \code{Ids} are the column number of \code{X} as it is inputed,
#' and not the name of the column. The indices are sorted by increasing number. Duplicates
#' are deleted. \code{Sds} may contain duplicates.
#'
#' @author Samuel Orso \email{Samuel.Orso@unige.ch}
#'
#' @references Guerrier, S., Mili, N., Molinari, R., Orso, S., Avella-Medina, M. and Ma, Y. (2015)
#' A Paradigmatic Regression Algorithm for Gene Selection Problems.
#' \emph{submitted manuscript}. \url{http://arxiv.org/abs/1511.07662}.
#'
#' @seealso \code{\link[panning]{CVmFold}}, \code{\link[panning]{GeneralStep}}
#'
#' @examples
#' \dontrun{
#' #####
#' # Simulate a logistic regression
#' n <- 50
#' set.seed(123)
#' beta <- c(1, rpois(40, lambda = 0.5))
#' p <- length(beta)
#' X <- matrix(rnorm((p-1)*n), nrow=n, ncol=(p-1))
#' y <- rbinom(n,1,1/(1+exp(-tcrossprod(beta, cbind(1, X)))))
#' #####
#'
#' # (can take several seconds to run)
#' IStep <- InitialStep(y = y, X = X, family = binomial(link = "logit"), type = "response",
#'                      divergence = "classification", trace = FALSE)
#'
#' # Run the parallelised version (4 cores)
#' IStep <- InitialStep(y = y, X = X, family = binomial(link = "logit"), type = "response",
#'                      divergence = "classification", proc = 2, trace = FALSE)
#' }
#' @importFrom doRNG %dorng%
#' @importFrom doParallel registerDoParallel
#' @importFrom parallel makeCluster stopCluster
#' @importFrom foreach "%dopar%" foreach
#' @export
InitialStep <- function(y, X, d = 1L, alpha = 0.05, B = NULL, seed = 951L, m = 10L, K = 10L, family,
                        type = NULL, divergence, W = NULL, proc = 1L, C0 = 0.5, increasing = FALSE,
                        trace = TRUE, ... ){
        # Initial values
        n <- length(y)          # number of observations
        p <- ncol(X)            # number of variables
        nm <- choose(p, d)      # number of different models of size d (usually d=1 for the initial step)
        if( is.null(B) ) B <- nm # explore all the nm models if B is not specified (!!! could be time consuming !!!)

        # Initialising cluster for parallelisation
        cl <- makeCluster( proc )
        registerDoParallel( cl )

        # Bootstrap procedure
        if( B < nm )
        {
                out.foreach <- foreach (i = 1:B, .combine = 'comb', .multicombine = T, .init=list(list(), list()),
                                        .export = 'CVmFold', .options.RNG = seed, .packages = 'nnet' ) %dorng% {
                                                rc <- sample.int( p, d )                                # Pick d variable(s) randomly
                                                Xboot <- X[,rc]                                         # Construct X matrix
                                                cv.error <- CVmFold( y=y, X=Xboot, m=m, K=K, W=W, family=family, type=type,
                                                                     divergence=divergence, C0=C0, increasing=increasing, trace=trace, ...  )      # Compute the m-fold-CV
                                                return( list( cv.error, rc ) )
                                        }
                cv.error <- unlist(out.foreach[[1]])
                var.mat <- matrix(unlist(out.foreach[[2]]),ncol=d,nrow=B)
        }else{
                if( d > 1 ){ var.mat <- t(combn(p,d)) }else{ var.mat <- matrix(seq_len(p),ncol=1,nrow=p) } # all combination of d among p variables

                i = 0 # Global Scoping issue?

                out.foreach <- foreach (i = 1:nm, .combine = c, .export = 'CVmFold', .packages = 'nnet' ) %dopar% {
                        rc <- var.mat[i,]                                       # Pick d variable(s) (non-randomly)
                        Xboot <- X[,rc]                                         # Construct X matrix
                        cv.error <- CVmFold( y=y, X=Xboot, m=m, K=K, W=W, family=family, type=type,
                                             divergence=divergence, C0=C0, increasing=increasing, trace=trace, ...  )      # Compute the m-fold-CV
                        return( cv.error )
                }
                cv.error <- out.foreach
        }

        stopCluster( cl )

        # Results
        q.alpha <- quantile(cv.error, probs = alpha)            # Compute the empirical alpha-quantiles
        Sds <- var.mat[cv.error <= q.alpha,]                    # Construct the set Sd*
        Ids <- sort.int( unique( c(Sds) ), method = 'quick')    # Construct the set Id* (sort in increasing order)

        out <- list( Ids=Ids, Sds=Sds, cv.error=cv.error, q.alpha=q.alpha, var.mat=var.mat )
        return(out)
}


#' General step of panning algorithm
#'
#' \code{GeneralStep} computes the intial step of the Panning Algorithm.
#'
#' This function computes the m-fold Cross-validation (CV) prediction error for \code{B} models
#' of size \code{d}. Each of those \code{B} models are randomly constructed with the following scheme:
#' a predictor has a probability \code{pi} to be selected from \code{Id_1s} and a probability
#' 1-\code{pi} from its complement; a predictor can appear at maximum once in one model (no replacement
#' within a model).
#'
#' The \code{seed} can be fixed for reproducibility.
#'
#' This function is computationnaly time consuming proportionally to the size of \code{B}.
#'
#' @param y,X,m,K,family,type,divergence,C0,W,increasing,trace,...   (see function \code{\link[panning]{CVmFold}})
#' @param d,alpha,B,seed,proc   (see function \code{\link[panning]{InitialStep}})
#' @param Id_1s                 is the set of indices of promising variables of model of size \code{d-1}.
#' @param pi                    is the probability of selecting a predictor from \code{Id_1s}.
#'
#' @return \code{GeneralStep} returns a list with the following components (exactly the same as in
#' \code{\link[panning]{InitialStep}}):
#' \describe{
#'      \item{\code{Ids}}{is the set \eqn{I_d^*} of indices of predictors with prediction errors
#'      \code{cv.error}<= \code{q.alpha}.}
#'      \item{\code{Sds}}{is the set \eqn{S_d^*} of models of size \code{d} with
#'      prediction errors \code{cv.error}<= \code{q.alpha}.}
#'      \item{\code{cv.error}}{is a (\code{B} x 1) vector of CV predictions errors.}
#'      \item{\code{q.alpha}}{is the empirical \code{alpha}-quantile computed on \code{cv.error}.}
#'      \item{\code{var.mat}}{is a (\code{B}x\code{d}) matrix of indices of the explored models.}
#' }
#' The indices returned by \code{Ids} are the column number of \code{X} as it is inputed,
#' and not the name of the column. The indices are sorted by increasing number. Duplicates
#' are deleted. \code{Sds} may contain duplicates.
#'
#' @author Samuel Orso \email{Samuel.Orso@unige.ch}
#'
#' @references Guerrier, S., Mili, N., Molinari, R., Orso, S., Avella-Medina, M. and Ma, Y. (2015)
#' A Paradigmatic Regression Algorithm for Gene Selection Problems.
#' \emph{submitted manuscript}. \url{http://arxiv.org/abs/1511.07662}.
#'
#' @seealso \code{\link[panning]{CVmFold}}, \code{\link[panning]{InitialStep}}
#'
#' @examples
#' \dontrun{
#' #####
#' # Simulate a logistic regression
#' n <- 50
#' set.seed(123)
#' beta <- c(1, rpois(40, lambda = 0.5))
#' p <- length(beta)
#' X <- matrix(rnorm((p-1)*n), nrow=n, ncol=(p-1))
#' y <- rbinom(n,1,1/(1+exp(-tcrossprod(beta, cbind(1, X)))))
#' #####
#' # Assume that Id_1s obtained from the Initial Step is
#' # (see example in \code{\link[panning]{InitialStep}})
#' Id_1s <- c(24,33)
#' # (can take several seconds to run)
#' GStep <- GeneralStep(y = y, X = X, Id_1s = c(24,33), d = 2, B = 50,
#'                      family = binomial(link = "logit"), type = "response",
#'                      divergence = "classification", trace = FALSE)
#'
#' # Run the parallelised version (4 cores)
#' GStep <- GeneralStep(y = y, X = X, Id_1s = c(24,33), d = 2, B = 50,
#'                      family = binomial(link = "logit"), type = "response",
#'                      divergence = "classification", proc = 2, trace = FALSE)
#' }
#' @export
GeneralStep <- function(y, X, Id_1s, pi = 0.5, B = 500L, d, alpha = 0.05, seed = 854751L,
                        K = 10L, m = 10L, family, type = NULL, divergence, W = NULL, proc = 1L,
                        C0 = 0.5, increasing = FALSE, trace = TRUE, ... ){
        # Initial values
        n <- length(y)          # number of observations
        p <- ncol(X)            # number of variables
        Id_1c <- seq_len(p)     # Construct the set Id_1c including variables not selected as promising in step d-1
        if( is.unsorted(Id_1s) ) Id_1s <- sort.int( Id_1s, method = 'quick')
        Id_1c <- Id_1c[-Id_1s]
        ps <- length(Id_1s)     # number of variables included in the promising set at step d-1
        pc <- length(Id_1c)     # number of variables not included in the promising set at step d-1

        # Initialising cluster for parallelisation
        cl <- makeCluster( proc )
        registerDoParallel( cl )

        # Bootstrap procedure
        out.foreach <- foreach (i = 1:B, .combine = 'comb', .multicombine = T, .init=list(list(), list()),
                                .export = 'CVmFold', .options.RNG = seed, .packages = 'nnet' ) %dorng% {
                                        # Important sampling
                                        rset <- rbinom( n = d, size = 1, prob = pi )    # selection of the sets
                                        nrset <- sum(rset)                              # number of variables to select from Id_1s
                                        if( nrset > ps ) nrset <- ps
                                        rc <- c( Id_1s[sample.int(ps, nrset)], Id_1c[sample.int(pc, d-nrset)] )  # Pick d variables

                                        # Computation of the m-fold-CV errors
                                        Xboot <- X[,rc]                                         # Construct X matrix
                                        cv.error <- CVmFold( y=y, X=Xboot, m=m, K=K, W=W, family=family, type=type,
                                                             divergence=divergence, C0=C0, increasing=increasing,
                                                             trace=trace, ...  )      # Compute the m-fold-CV
                                        return( list( cv.error, rc ) )
                                }
        cv.error <- unlist(out.foreach[[1]])
        var.mat <- matrix(unlist(out.foreach[[2]]),ncol=d,nrow=B)

        stopCluster( cl )

        # Results
        q.alpha <- quantile(cv.error, probs = alpha)    # Compute the empirical alpha-quantiles
        Sds <- var.mat[cv.error <= q.alpha,]                    # Construct the set Sd*
        Ids <- sort.int( unique( c(Sds) ), method = 'quick')    # Construct the set Id*

        out <- list( Ids=Ids, Sds=Sds, cv.error=cv.error, q.alpha=q.alpha, var.mat=var.mat )
        return(out)
}

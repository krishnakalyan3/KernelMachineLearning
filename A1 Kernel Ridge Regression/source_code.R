# Author : Sai Krishna Kalyan
# Date : 20 Sept 2016

library(ggplot2)

# Set the seed to make the expriment reproduceable
set.seed(6046)

# As per the assignment generate data 
A <- 1052
x <- seq(0.1, 100, length.out = A)
a = 10
b = 50
c = 80

# Function t
t = 0.5 * sin(x - a)/(x - a) + 
  0.8 * sin(x - b)/(x - b) + 
  0.3 * sin(x - c)/(x - c) + rnorm(x, sd=0.05)

# Inspect data
head(x)
awesome_data = data.frame(x,t)
names(awesome_data) = c('x', 't')

# Experminet with polynomial regression
ggplot(data=awesome_data, aes(x, t, color="standard_fn")) +
  geom_line(color="black") +
  geom_point(size = .1) +
  geom_smooth(method = "lm", formula = y ~ x, size =0.5, aes(colour="poly_1")) +
  geom_smooth(method = "lm", formula = t ~ poly(x,3), size =0.5, aes(colour="poly_3")) +
  geom_smooth(method = "lm", formula = t ~ poly(x,10), size =0.5, aes(colour="poly_10")) + 
  geom_smooth(method = "lm", formula = t ~ poly(x,21), size =0.5, aes(colour="poly_21")) +
  scale_colour_manual(name="Legend",
                      values=c(poly_1="#FF0000", poly_3="#FF9900", poly_10="#CBFF19", poly_21="#32FF65",standard_fn="black")) +
  theme_bw() +
  guides(colour = guide_legend(override.aes = list(size=1))) +
  ggtitle("Polynomial Regression Fit") 

# RBF Kernel
sigma = 1
N = length(x)
lambda = 0.01
kk = tcrossprod(x)
dd = diag(kk)
myRBF.kernel = exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))
ident.N = diag(rep(1,N))
alphas = solve(myRBF.kernel + lambda*ident.N)
alphas = alphas %*% t
kernel1 = myRBF.kernel %*% alphas

lambda = 10
RBF.kernel = exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))
alphas = solve(RBF.kernel + lambda*ident.N)
alphas = alphas %*% t
kernel2 = RBF.kernel %*% alphas

lambda = 100
RBF.kernel = exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))
alphas = solve(RBF.kernel + lambda*ident.N)
alphas = alphas %*% t
kernel3 = RBF.kernel %*% alphas

# Experiment with differnet lambda values 0.01, 10, 100
ggplot(data=awesome_data, aes(x, t, color="standard_fn")) +
  geom_line(color="black") +
  geom_point(size = 0.1) +
  geom_point(aes(x = x, y = kernel1, color="lambda_.01"), size = 0.01) +
  geom_point(aes(x = x, y = kernel2, color="lambda_10"), size = 0.01) +
  geom_point(aes(x = x, y = kernel3, color="lambda_100"), size = 0.01) +
  scale_colour_manual(name="Lambda Values",
                      values=c(standard_fn="black", lambda_100="#FFFF00", lambda_10="#00FF7F", lambda_.01="#4169E1")) +
  theme_bw() +
  guides(colour = guide_legend(override.aes = list(size=1))) +
  ggtitle("RBF kernels with constant Sigma")

lambda = 0.01
sigma = 0.0005
RBF.kernel = exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))
alphas = solve(RBF.kernel + lambda*ident.N)
alphas = alphas %*% t
kernel4 = RBF.kernel %*% alphas

sigma = 0.005
RBF.kernel = exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))
alphas = solve(RBF.kernel + lambda*ident.N)
alphas = alphas %*% t
kernel5 = RBF.kernel %*% alphas

sigma = 0.5
RBF.kernel = exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))
alphas = solve(RBF.kernel + lambda*ident.N)
alphas = alphas %*% t
kernel6 = RBF.kernel %*% alphas

# Experimenting with different sigma values 0.005, 0.005, 0.5
ggplot(data=awesome_data, aes(x, t, color="standard_fn")) +
  geom_line(color="black") +
  geom_point(size = 0.1) +
  geom_point(aes(x = x, y = kernel4, color="sigma_0.0005"), size = 0.01) +
  geom_point(aes(x = x, y = kernel5, color="sigma_0.005"), size = 0.01) +
  geom_point(aes(x = x, y = kernel6, color="sigma_0.5"), size = 0.01) +
  scale_colour_manual(name="Sigma Values",
                      values=c(standard_fn="black", sigma_0.5="#FFB000", sigma_0.005="#9CFF30", sigma_0.0005="#05F386")) +
  theme_bw() +
  guides(colour = guide_legend(override.aes = list(size=1))) +
  ggtitle("RBF kernels with constant Lambda")
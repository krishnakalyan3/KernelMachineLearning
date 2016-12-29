# Multiple Correspondence Analysis (MCA)
# By T. Aluja

acm <- function (dact,dsup=0,nd=0)
{
  library(MASS)
  
  Sup = 1
  if (length(dim(dsup)) == 0 ) Sup = 0
  n <- dim(dact)[1]
  pact <- dim(dact)[2]
  if (Sup == 1) psup <- dim(dsup)[2]
  nmod = 0
  for (j in 1:pact) {nmod = nmod + length(levels(dact[,j]))}
  if (nd==0) nd = nmod - pact
  if (nd > n-1) nd = n-1
  
  # MCA properly
  dm <- mca(dact,nd)
  
  # get coordinates of observations
  Psi <- scale(dm$rs) %*% diag(dm$d)
  
  # get coordinates of active modalities
  
  Fi <- matrix(data=NA,nrow=nmod,ncol=nd)
  nco <- 0
  for (j in 1:pact) 
  {
    ini = nco+1
    nco = nco+length(levels(dact[,j]));
    ifi = nco;
    for (a in 1:nd) {Fi[ini:ifi,a] = tapply(Psi[,a],dact[,j],mean)}
  }
  Fi = data.frame(Fi);
  row.names(Fi) = dimnames(dm$cs)[[1]]
  
  # get coordinates of illustrative modalities
  
  Fs = NULL
  if ( Sup == 1 ) {
    nsup = 0
    for (j in 1:psup) {nsup = nsup + length(levels(dsup[,j]))}        
    Fs <- matrix(data=NA,nrow=nsup,ncol=nd)
    nco <- 0
    etiq <- NULL
    for (j in 1:psup) 
    {
      ini = nco+1
      nco = nco+length(levels(dsup[,j]));
      ifi = nco;
      etiq = c(etiq,paste(names(dsup)[j],levels(dsup[,j])));
      for (a in 1:nd) {Fs[ini:ifi,a] = tapply(Psi[,a],dsup[,j],mean)}}
    Fs = data.frame(Fs);
    row.names(Fs) = etiq 
  }
  list(rs=Psi, cs=Fi, csup=Fs, vaps=dm$d^2)
}

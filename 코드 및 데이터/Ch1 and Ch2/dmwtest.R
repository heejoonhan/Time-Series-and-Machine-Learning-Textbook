dmwtest<-function(real,f1,f2){

se1=abs(real-f1)  #MAE loss function from the first model
se2=abs(real-f2)  #MAE loss function from the second model

# may try (real-f1)^2 for MSE

dt = se1 - se2   # forecast loss differential, 

# DMW test #
nf = length(real)

e=dt-mean(dt);
s0=t(e)%*%e/nf;
q=round(nf^(1/3)); #bandwidth
s2=0;
L=1;

while (L<=q){
  s1=0;
  t=L+1;
  while (t<=nf){
    sloop=(1-L/(q+1))*e[t]*e[t-L];
    s1=s1+sloop;
    t=t+1;
  } 
  s2=s2+s1/nf;
  L=L+1;
}

varcov=s0+2*s2;
se=sqrt(varcov);

DMW=as.numeric(sqrt(nf)*mean(dt)/se);   # DMW test statistic
mae1 = mean(se1);
mae2 = mean(se2);


PVAL <- as.numeric(2 * pnorm(-abs(DMW)))

structure(list(statistic = DMW, p.value = PVAL, MAE1 = mae1, MAE2 = mae2 ))

}

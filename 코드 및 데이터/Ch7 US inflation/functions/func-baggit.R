
baggit=function (mat, pre.testing = "joint", fixed.controls = NULL, 
                 t.stat = 1.96,ngroups=10) 
{
  y = mat[, 1]
  X = mat[, -1]
  
  df = data.frame(y = y, X = X)
  
  if (pre.testing == "joint") {
    if (nrow(X) < ncol(X)) {
      stop("Error: Type = joint is only for data with more observations than variables")
    }
    m1 = lm(y ~ ., data = df)
    t1 = summary(m1)$coefficients[-1, 3]
    s1 = which(abs(t1) > t.stat)
    if (length(s1) == 0) {
      stop("Error: The pre-testing excluded all variables", 
           "/n")
    }
  }
  if (pre.testing == "group-joint") {
    
    
    N=ncol(X)
    n=ceiling(N/ngroups)
    varind=1:N
    t1=rep(NA,N)
    for(i in 1:ngroups){
      selected=sample(order(varind),min(n,length(varind)),replace = FALSE)
      X0=X[,varind[selected]]
      df_group = data.frame(y=y, X=X0)
      m1=lm(y~.,data=df_group)
      
      t0=rep(0,length(selected))
      aux=which(is.na(coef(m1)[-1]))
      t = summary(m1)$coefficients[-1, 3]
      if(length(aux)==0){
        t0=t
      }else{
        t0[-aux]=t
      }
      
      t1[varind[selected]]=t0
      varind=varind[-selected]
    }
    
    s1 = which(abs(t1) > t.stat)
    if (length(s1) == 0) {
      stop("Error: The pre-testing excluded all variables", 
           "/n")
    }
  }
  if (pre.testing == "individual") {
    if (length(fixed.controls) > 0) {
      w = X[, fixed.controls]
      nonw = setdiff(1:ncol(X), fixed.controls)
      df = data.frame(y = y, w = w)
    }
    else {
      w = rep(0, nrow(X))
      nonw = 1:ncol(X)
    }
    store.t = rep(NA, ncol(X))
    store.t[fixed.controls] = Inf
    for (i in nonw) {
      m1 = lm(y ~ X[, i] + .,data = df)
      t1 = summary(m1)$coefficients[2, 3]
      store.t[i] = t1
    }
    s1 = which(abs(store.t) > t.stat)
  }
  if (length(s1) > nrow(X)) {
    stop("Error: The pre-testing was not able to reduce the dimension to N<T")
  }
  df_final = data.frame(y = y, X = X[, s1])
  
  m2 = lm(y ~ ., data = df_final)
  final.coef = rep(0, ncol(X))
  final.coef[s1] = coef(m2)[-1]
  names(final.coef) = colnames(X)
  final.coef = c(coef(m2)[1], final.coef)
  return(final.coef)
}

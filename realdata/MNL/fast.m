Path = "A2C2023-02-18-17-42-55/mat" ;


%p = parpool('local',4);
p.IdleTimeout = Inf
opts = parforOptions(p,"MaxNumWorkers",4)
parfor (i= 1:4,opts)
    F=Path+"/type"+num2str(i)+".mat";
    Data = load(F);
    data=Data.data;
    beta=rand(6,1);
    options =optimoptions('fminunc','Display','iter-detailed','MaxIterations',10,'OptimalityTolerance',0.001);
    beta=fminunc(@(beta)cal_MNL_LL(beta,data),beta,options)
    mnl_val = cal_MNL_LL(beta,data)
    parsave(i, beta)

    end
    

    
  
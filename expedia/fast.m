%B = "realdata_beta.mat";
%Beta = load(B)
%beta = Beta.beta
F="real_data.mat";
Data = load(F);
data=Data.data;
fea = length(data(1,4:13));

[train_data, test_data] = split_data(data,0.8);

beta=rand(fea,1);
options =optimoptions('fminunc','Display','iter-detailed','MaxIterations',40,'OptimalityTolerance',0.0001);
beta=fminunc(@(beta)cal_MNL_LL(beta,train_data),beta,options)

mnl_val = cal_MNL_LL(beta,test_data)
save("realdata_beta.mat","beta")
    
  
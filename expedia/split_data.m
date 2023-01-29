function [train_data, test_data] = split_data(data,p)
    userId=unique(data(:,1));
    kk = size(userId, 1);
    n=randperm(size(userId,1));
    
    %n = n(1:5000);
    %kk = 5000;
    
    Train_Id = userId(n(1:ceil(kk*p)));
    Test_Id = userId(n(ceil(kk*p)+1:end));
    train_data = data(ismember(data(:,1),Train_Id),:);
    test_data = data(ismember(data(:,1),Test_Id),:);
end


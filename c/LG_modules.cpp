#include<torch/torch.h>
#include<torch/extension.h>
#include<iostream>
#include<cstdlib>
#include<ctime>


torch::Tensor cal_idx(const torch::Tensor& xyz, const int nsample){
    AT_ASSERTM(xyz.dim() ==2, "xyz dim must be 2");
    int N = xyz.size(0);
    torch::Tensor idx = torch::zeros((nsample));

    srand((int)time(0));
    torch::Tensor tmp = torch::ones(N)*1e10;
    int farthest_point = rand() % (N);

    torch::Tensor dist, mask;

    idx[0] = farthest_point;
    dist = torch::sum(pow((xyz - xyz[farthest_point]),2),-1);
    tmp = tmp.type_as(dist);


    for(int i=0;i<nsample;i++){
        idx[i] = farthest_point;
        dist = torch::sum(pow((xyz - xyz[farthest_point]),2),-1);
//        tmp = tmp.type_as(dist);
        tmp = at::min(tmp,dist);

        farthest_point = at::argmax(tmp,-1).template item<int>();
    }
    return idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("cal_idx", &cal_idx, "LG cal_idx");
}

# Stitching-project-part-2

This so far we are stitching ResNet-50 models together. We train our intial two models in the code training_models, we then stitching together at 16 different points in the stitching_models code, we finally anaylse the models in analyse_models. For the analysis we measure their performance and then the simplicity by (currently) seven metrics: gradient norm, loss sharpness, compressed bit size, total parameters, number of non-zero parameters, Hessian top eigenvalue, Hessian trace.

//
// Created by Kelly Yung on 2021/9/23.
//

#include "FedTree/FL/FLtrainer.h"
#include "FedTree/FL/FLparam.h"
#include "FedTree/Tree/GBDTparam.h"
#include "FedTree/FL/partition.h"
#include "FedTree/predictor.h"
#include "FedTree/parser.h"


extern "C" {

    void set_logger(int verbose) {
        if(verbose == 0) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
        }
        else if (verbose == 1) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        }
    }

    void fit (int n_parties, int partition, float alpha, int n_hori, int n_verti, char *mode, char *partition_mode,
              char *privacy_tech, char *propose_split, char *merge_histogram, float variance, float privacy_budget,
              int depth, int n_trees, float min_child_weight, float lambda, float gamma, float column_sampling_rate,
              int verbose, int bagging, int n_parallel_trees, float learning_rate, char *objective, int* num_class,
              int n_device, int max_num_bin, int seed, float ins_bagging_fraction, int reorder_label, float constant_h,
              // DataSet info
              int row_size, float *val, int *row_ptr, int *col_ptr, float *label,
              // Tree info
              char *tree_method, Tree *&model, int *tree_per_iter, float *group_label,
              int *group, int num_group=0
              ) {
        LOG(INFO) << "Start training";

        // Initialize model params
        LOG(INFO) << "Initialize FL parameters";
        FLParam fl_param;
        fl_param.n_parties = n_parties;
        fl_param.partition = partition == 1? true : false;
        fl_param.alpha = alpha;
        fl_param.n_hori = n_hori;
        fl_param.n_verti = n_verti;
        fl_param.mode = mode;
        fl_param.partition_mode = partition_mode;
        fl_param.privacy_tech = privacy_tech;
        fl_param.propose_split = propose_split;
        fl_param.merge_histogram = merge_histogram;
        fl_param.variance = variance;
        fl_param.privacy_budget = privacy_budget;
        fl_param.seed = seed;
        fl_param.ins_bagging_fraction = ins_bagging_fraction;

        GBDTParam gbdt_param;
        gbdt_param.depth = depth;
        gbdt_param.n_trees = n_trees;
        gbdt_param.min_child_weight = min_child_weight;
        gbdt_param.lambda = lambda;
        gbdt_param.gamma = gamma;
        gbdt_param.column_sampling_rate = column_sampling_rate;
        gbdt_param.verbose = verbose;
        gbdt_param.bagging = bagging == 1? true : false;
        gbdt_param.n_parallel_trees = n_parallel_trees;
        gbdt_param.learning_rate = learning_rate;
        gbdt_param.objective = objective;
        gbdt_param.num_class = num_class[0];
        gbdt_param.tree_method = "hist";
        gbdt_param.n_device = n_device;
        gbdt_param.tree_per_round = 1;
        gbdt_param.max_num_bin = max_num_bin;
        gbdt_param.rt_eps = 1e-6;
        gbdt_param.metric = "default";
        gbdt_param.reorder_label = reorder_label;
        gbdt_param.constant_h = constant_h;

        if (fl_param.privacy_tech == "dp" && gbdt_param.constant_h == 0)
            gbdt_param.constant_h = 1.0;

        if (fl_param.n_hori == -1){
            if(fl_param.mode == "horizontal"){
                fl_param.n_hori = fl_param.n_parties;
            }
            else
                fl_param.n_hori = 1;
        }
        if (fl_param.n_verti == -1){
            if(fl_param.mode == "vertical"){
                fl_param.n_verti = fl_param.n_parties;
            }
            else
                fl_param.n_verti = 1;
        }

        set_logger(verbose);
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");

        LOG(INFO) << "Load Sparse Data to Training Set";
        DataSet dataset;
        dataset.load_from_sparse(row_size, val, row_ptr, col_ptr, label, group, num_group, gbdt_param);
        num_class[0] = gbdt_param.num_class;
        fl_param.gbdt_param = gbdt_param;
        LOG(INFO) << "Partition the Data";
        // Partition the dataset
        n_parties = fl_param.n_parties;
        vector<DataSet> subsets(n_parties);
        vector<SyncArray<bool>> feature_map(n_parties);
        std::map<int, vector<int>> batch_idxs;
        Partition partitioning;
        if (fl_param.partition == true) {
            if (fl_param.partition_mode == "hybrid") {
                if (fl_param.mode == "horizontal")
                    CHECK_EQ(fl_param.n_verti, 1);
                if (fl_param.mode == "vertical")
                    CHECK_EQ(fl_param.n_hori, 1);
                partitioning.horizontal_vertical_dir_partition(dataset, n_parties, fl_param.alpha, feature_map, subsets,
                                                            fl_param.n_hori, fl_param.n_verti);
            }
            else if (fl_param.partition_mode == "vertical") {
                dataset.csr_to_csc();
                partitioning.homo_partition(dataset, n_parties, false, subsets, batch_idxs, fl_param.seed);
            }else if (fl_param.partition_mode == "horizontal") {
                dataset.csr_to_csc();
                partitioning.homo_partition(dataset, n_parties, true, subsets, batch_idxs, fl_param.seed);
            }
        }
        GBDTParam &param = fl_param.gbdt_param;
        //correct the number of classes
        if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos || param.metric == "error") {
            int num_class;
            if(fl_param.partition) {
                num_class = dataset.label.size();
                if ((param.num_class == 1) && (param.num_class != num_class)) {
                    LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
                    param.num_class = num_class;
                }
            }
            if(param.num_class > 2)
                param.tree_per_round = param.num_class;
        }
        else if(param.objective.find("reg:") != std::string::npos){
            param.num_class = 1;
        }
        vector<Party> parties(n_parties);
        vector<int> n_instances_per_party(n_parties);
        Server server;
        if(fl_param.mode != "centralized") {
            LOG(INFO) << "initialize parties";
            for (int i = 0; i < n_parties; i++) {
                if(fl_param.mode == "vertical")
                    parties[i].vertical_init(i, subsets[i], fl_param);
                else if(fl_param.mode == "horizontal" || fl_param.mode == "ensemble")
                    parties[i].init(i, subsets[i], fl_param);
                n_instances_per_party[i] = subsets[i].n_instances();
            }
            LOG(INFO) << "initialize server";
            if (fl_param.mode == "vertical") {
                if(fl_param.partition)
                    server.vertical_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset.y, dataset.label);
                else
                    server.vertical_init(fl_param, subsets[0].n_instances(), n_instances_per_party, subsets[0].y, subsets[0].label);
            } else if (fl_param.mode == "horizontal" || fl_param.mode == "ensemble") {
                server.horizontal_init(fl_param);
            }
        }

//        // Initialize server
//        LOG(INFO) << "initialize server";
//        Server server;
//        if (fl_param.mode == "vertical") {
//            server.vertical_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset.y, dataset.label);
//        }else if (fl_param.mode == "horizontal") {
//            server.horizontal_init(fl_param);
//        }

        LOG(INFO) << "Run the trainer";
        // Run different training methods based on mode
        FLtrainer trainer;
        if (param.tree_method == "auto")
            param.tree_method = "hist";
        else if (param.tree_method != "hist"){
            LOG(INFO)<<"FedTree only supports histogram-based training yet";
            exit(1);
        }
        if(fl_param.mode == "hybrid") {
            trainer.hybrid_fl_trainer(parties, server, fl_param);
        } else if(fl_param.mode == "ensemble") {
            trainer.ensemble_trainer(parties, server, fl_param);
        }else if(fl_param.mode == "solo") {
            trainer.solo_trainer(parties, fl_param);
        }else if(fl_param.mode == "centralized") {
            GBDT gbdt;
            gbdt.train(fl_param.gbdt_param, dataset);
        }else if (fl_param.mode == "vertical") {
            trainer.vertical_fl_trainer(parties, server, fl_param);
        }else if (fl_param.mode == "horizontal") {
            trainer.horizontal_fl_trainer(parties, server, fl_param);
        }

        // Return boosted model
        vector<vector<Tree>> boosted_model = server.global_trees.trees;
        *tree_per_iter = (int)(boosted_model[0].size());
        model = new Tree[n_trees * (*tree_per_iter)];
        for(int i = 0; i < n_trees; i++) {
            for(int j = 0; j < *tree_per_iter; j++){
                model[i * (*tree_per_iter) + j] = boosted_model[i][j];
            }
        }
        for (int i = 0; i < dataset.label.size(); ++i) {
            group_label[i] = dataset.label[i];
        }
    }

    void predict(int row_size, float *val, int *row_ptr, int *col_ptr, float *y_pred, Tree *&model,
             int n_trees, int trees_per_iter, char *objective, int num_class, float learning_rate, float *group_label,
             int *group, int num_group=0, int verbose=1, int bagging=0) {
        //load model
        GBDTParam model_param;
        model_param.objective = objective;
        model_param.learning_rate = learning_rate;
        model_param.num_class = num_class;
        model_param.bagging = bagging;
        DataSet test_dataset;
        test_dataset.load_from_sparse(row_size, val, row_ptr, col_ptr, NULL, group, num_group, model_param);
        set_logger(verbose);
        if(group_label != NULL) {
            test_dataset.label.clear();
            for (int i = 0; i < num_class; ++i) {
                test_dataset.label.emplace_back(group_label[i]);
            }
        }
        else{
            for (int i = 0; i < num_class; ++i) {
                test_dataset.label.emplace_back(i);
            }
        }
        // predict
        SyncArray<float_type> y_predict;
        vector<vector<Tree>> boosted_model_in_mem;
        for(int i = 0; i < n_trees; i++){
            boosted_model_in_mem.push_back(vector<Tree>());
            CHECK_NE(model, NULL) << "model is null!";
            for(int j = 0; j < trees_per_iter; j++) {
                boosted_model_in_mem[i].push_back(model[i * trees_per_iter + j]);
            }
        }
        GBDT gbdt_tree(boosted_model_in_mem);
        gbdt_tree.predict_raw(model_param, test_dataset, y_predict);
        //convert the aggregated values to labels, probabilities or ranking scores.
        std::unique_ptr<ObjectiveFunction> obj;
        obj.reset(ObjectiveFunction::create(model_param.objective));
        obj->configure(model_param, test_dataset);
        obj->predict_transform(y_predict);
        vector<float_type> y_pred_vec(y_predict.size());
        memcpy(y_pred_vec.data(), y_predict.host_data(), sizeof(float_type) * y_predict.size());
        for(int i = 0; i < y_pred_vec.size(); i++) {
            y_pred[i] = y_pred_vec[i];
        }//float_type may be double/float, so convert to float for both cases.
    }

    void predict_proba(int row_size, float *val, int *row_ptr, int *col_ptr, float *y_pred, Tree *&model,
                 int n_trees, int trees_per_iter, char *objective, int num_class, float learning_rate, float *group_label,
                 int *group, int num_group=0, int verbose=1, int bagging=0) {
        //load model
        GBDTParam model_param;
        model_param.objective = objective;
        model_param.learning_rate = learning_rate;
        model_param.num_class = num_class;
        model_param.bagging = bagging;
        if(model_param.objective == "multi:softmax"){
            model_param.objective = "multi:softprob";
        }
        DataSet test_dataset;
        test_dataset.load_from_sparse(row_size, val, row_ptr, col_ptr, NULL, group, num_group, model_param);
        set_logger(verbose);
        test_dataset.label.clear();
        if(group_label != NULL) {
            test_dataset.label.clear();
            for (int i = 0; i < num_class; ++i) {
                test_dataset.label.emplace_back(group_label[i]);
            }
        }
        else{
            for (int i = 0; i < num_class; ++i) {
                test_dataset.label.emplace_back(i);
            }
        }
        // predict
        SyncArray<float_type> y_predict;
        vector<vector<Tree>> boosted_model_in_mem;
        for(int i = 0; i < n_trees; i++){
            boosted_model_in_mem.push_back(vector<Tree>());
            CHECK_NE(model, NULL) << "model is null!";
            for(int j = 0; j < trees_per_iter; j++) {
                boosted_model_in_mem[i].push_back(model[i * trees_per_iter + j]);
            }
        }
        GBDT gbdt_tree(boosted_model_in_mem);
        gbdt_tree.predict_raw(model_param, test_dataset, y_predict);
        //convert the aggregated values to labels, probabilities or ranking scores.
        std::unique_ptr<ObjectiveFunction> obj;
        obj.reset(ObjectiveFunction::create(model_param.objective));
        obj->configure(model_param, test_dataset);
        if(model_param.objective == "multi:softprob"){
            obj->predict_transform(y_predict);
        }
        vector<float_type> y_pred_vec(y_predict.size());
        memcpy(y_pred_vec.data(), y_predict.host_data(), sizeof(float_type) * y_predict.size());
        for(int i = 0; i < y_pred_vec.size(); i++) {
            y_pred[i] = y_pred_vec[i];
        }//float_type may be double/float, so convert to float for both cases.
    }

    void save_model(char *model_path, char *objective, float learning_rate, int num_class, int n_trees,
              int trees_per_iter, Tree *&model, float *group_label){

        GBDTParam model_param;
        model_param.objective = objective;
        model_param.learning_rate = learning_rate;
        model_param.num_class = num_class;
        model_param.n_trees = n_trees;
        vector<vector<Tree>> boosted_model;
        for(int i = 0; i < n_trees; i++){
            boosted_model.push_back(vector<Tree>());
            CHECK_NE(model, NULL) << "model is null!";
            for(int j = 0; j < trees_per_iter; j++) {
                boosted_model[i].push_back(model[i * trees_per_iter + j]);
            }
        }
        Parser parser;
        parser.save_model(model_path, model_param, boosted_model);
    }
    void load_model(char *model_path, float *learning_rate, int *num_class, int *n_trees,
                    int *trees_per_iter, char *objective, Tree *&model){
        GBDTParam model_param;
        vector<vector<Tree>> boosted_model;
        Parser parser;
        parser.load_model(model_path, model_param, boosted_model);
        *learning_rate = model_param.learning_rate;
        *num_class = model_param.num_class;
        *n_trees = model_param.n_trees;
        strcpy(objective, model_param.objective.c_str());
        *trees_per_iter = (int)(boosted_model[0].size());

        model = new Tree[*n_trees * (*trees_per_iter)];
        CHECK_EQ(*n_trees, boosted_model.size()) << n_trees << " v.s. " << boosted_model.size();
        for(int i = 0; i < *n_trees; i++)
        {
            for(int j = 0; j < *trees_per_iter; j++){
                model[i * (*trees_per_iter) + j] = boosted_model[i][j];
            }
        }

    }

    void model_free(Tree* &model){
        if(model){
            delete []model;
        }
    }


    void centralize_train_a_subtree (float variance, float privacy_budget, int depth, int n_trees, float min_child_weight,
                                     float lambda, float gamma, float column_sampling_rate, int verbose, int bagging,
                                     int n_parallel_trees, float learning_rate, char *objective, int* num_class,
                                     int n_device, int max_num_bin, int seed, float ins_bagging_fraction,
                                     int reorder_label, float constant_h,
                                     int row_size, float *val, int *row_ptr, int *col_ptr, float *label,
                                     char *tree_method, Tree *&model, int *tree_per_iter, float *group_label,
                                     int *group, int num_group, int n_layer, int *insid_list, int *n_ins_list,
                                     float *gradient_g_list, float *gradient_h_list, int *n_node, float *input_gradient_g, float *input_gradient_h) {

        LOG(INFO) << "Start training";

        // Initialize model params

        GBDTParam gbdt_param;
        gbdt_param.depth = n_layer;
        gbdt_param.n_trees = n_trees;
        gbdt_param.min_child_weight = min_child_weight;
        gbdt_param.lambda = lambda;
        gbdt_param.gamma = gamma;
        gbdt_param.column_sampling_rate = column_sampling_rate;
        gbdt_param.verbose = verbose;
        gbdt_param.bagging = bagging == 1? true : false;
        gbdt_param.n_parallel_trees = n_parallel_trees;
        gbdt_param.learning_rate = learning_rate;
        gbdt_param.objective = objective;
        gbdt_param.num_class = num_class[0];
        gbdt_param.tree_method = "hist";
        gbdt_param.n_device = n_device;
        gbdt_param.tree_per_round = 1;
        gbdt_param.max_num_bin = max_num_bin;
        gbdt_param.rt_eps = 1e-6;
        gbdt_param.metric = "default";
        gbdt_param.reorder_label = reorder_label;
        gbdt_param.constant_h = constant_h;

        set_logger(verbose);
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");

        LOG(INFO) << "Load Sparse Data to Training Set";
        DataSet dataset;
        dataset.load_from_sparse(row_size, val, row_ptr, col_ptr, label, group, num_group, gbdt_param);
        num_class[0] = gbdt_param.num_class;

        FLParam fl_param;
        fl_param.gbdt_param = gbdt_param;

        GBDTParam &param = fl_param.gbdt_param;
        //correct the number of classes
        if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos || param.metric == "error") {
            if(param.num_class > 2)
                param.tree_per_round = param.num_class;
        }
        else if(param.objective.find("reg:") != std::string::npos){
            param.num_class = 1;
        }

//        // Initialize server
//        LOG(INFO) << "initialize server";
//        Server server;
//        if (fl_param.mode == "vertical") {
//            server.vertical_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset.y, dataset.label);
//        }else if (fl_param.mode == "horizontal") {
//            server.horizontal_init(fl_param);
//        }

        // Run different training methods based on mode

        if (param.tree_method == "auto")
            param.tree_method = "hist";
        else if (param.tree_method != "hist"){
            LOG(INFO)<<"FedTree only supports histogram-based training yet";
            exit(1);
        }

        GBDT gbdt;
        std::cout<<"before train a subtree"<<std::endl;

        gbdt.train_a_subtree(fl_param.gbdt_param, dataset, n_layer, insid_list, n_ins_list, gradient_g_list, gradient_h_list, n_node, input_gradient_g, input_gradient_h);
        std::cout<<"gbdt trees:"<<gbdt.trees.size()<<std::endl;

        vector<vector<Tree>> boosted_model = gbdt.trees;

        std::cout<<"boossted_model size:"<<boosted_model[0].size()<<std::endl;
        std::cout<<"n_trees:"<<n_trees<<std::endl;

        // Return boosted model
//        vector<vector<Tree>> boosted_model = server.global_trees.trees;
        *tree_per_iter = (int)(boosted_model[0].size());
        n_trees = 1;
        model = new Tree[n_trees * (*tree_per_iter)];

        for(int i = 0; i < n_trees; i++) {
            for(int j = 0; j < *tree_per_iter; j++){
                model[i * (*tree_per_iter) + j] = boosted_model[i][j];
            }
        }

        for (int i = 0; i < dataset.label.size(); ++i) {
            group_label[i] = dataset.label[i];
        }
        LOG(INFO)<<"finish scikit centralize train a subtree";
    }

    void update_a_layer_with_flag(int *flag_val, int *id_list, int *nins_list, float *gradient_g, float *gradient_h,
                                  int n_node, float lambda, float *leaf_val){
        //todo: encrypted g and h
        vector<int> n_ins_accu(n_node+1);
        n_ins_accu[0] = 0;
        for(int i = 1; i <= n_node; i++){
            n_ins_accu[i] = n_ins_accu[i-1];
            n_ins_accu[i] += nins_list[i-1];

        }

        for(int i = 0; i < n_node; i++){
            int n_ins = nins_list[i];
            float left_g = 0;
            float left_h = 0;
            float right_g = 0;
            float right_h = 0;
            int n_left = 0;
            int n_right = 0;
            for(int j = n_ins_accu[i]; j < n_ins_accu[i+1]; j++){
                if(flag_val[id_list[j]] != 0){
                    left_g += gradient_g[j];
                    left_h += gradient_h[j];
                    n_left++;
                }
                else{
                    right_g += gradient_g[j];
                    right_h += gradient_h[j];
                    n_right++;
                }
            }
            leaf_val[i*2] = -left_g / (left_h + lambda);
            leaf_val[i*2+1] = -right_h / (right_h + lambda);
        }
    }

}

function dict = construct_dict(train_image_paths, num_words)

%run('vlfeat/toolbox/vl_setup')
run ../vlfeat-0.9.21/toolbox/vl_setup
data_path = '../data/'; 
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', ...
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', ...
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'};
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub', ...
    'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For'};
    
num_train_per_cat = 50/10; % you can use smaller size of data set for debugging

fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels]=get_image_paths(data_path, categories, num_train_per_cat);  

descriptors = [];
num = size(train_image_paths,1);
for i = 1:num
    in_image = imread(train_image_paths{i});
    in_image = single(in_image);
    [~, descriptor] = vl_dsift(in_image); % vl_sift(in_image);
    descriptors = [descriptors descriptor];
end

[dict,~] = vl_kmeans(single(descriptors), num_words);

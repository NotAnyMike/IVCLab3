%% Preparation
run('vlfeat/toolbox/vl_setup') % Set up VL feat package.
data_path = '../data/'; 
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', ...
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', ...
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'};
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub', ...
    'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For'};

num_train_per_cat = 50; % you can use smaller size of data set for debugging

fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels]=get_image_paths(data_path, categories, num_train_per_cat);  

%% Step 1: Represent each image with a feature
fprintf('Using Tiny images representation for images\n') 

train_image = tiny_images(train_image_paths);
test_image  = tiny_images(test_image_paths);

%% Step 2: Classify each test image by training and using the appropriate classifier
fprintf('Using Nearest Neighbor classifier to predict test set categories\n');
k=5; % the K nearest neighbor
predicted_labels = nearest_neighbor(train_image, train_labels, test_image,k);  

%% print out the result
create_results_webpage( train_image_paths,test_image_paths,train_labels,test_labels,categories,abbr_categories,predicted_labels);

function predicted = nearest_neighbor(train_imgs, train_labels, test_imgs, k)
    D = pdist2(test_imgs,train_imgs);
    D_sorted = sort(D,2);
    k_closest = strings(size(train_imgs,1),k);
    predicted = strings(size(train_imgs,1),1);
    
    for i=1:size(test_imgs)
        for j=1:k
            k_closest(i,j)=train_labels(find(D(i,:)==D_sorted(i,j)));
        end
        predicted(i) = get_mode_str(k_closest(i,:));
    end
end

function mode_val = get_mode_str(x)
    y = unique(x);
    n = zeros(length(y), 1);
    for iy = 1:length(y)
      n(iy) = length(find(strcmp(y{iy}, x)));
    end
    [~, itemp] = max(n);
    mode_val = y(itemp);
end
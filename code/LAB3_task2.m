%% preparation

run('vlfeat/toolbox/vl_setup')
data_path = '../data/'; 
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', ...
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', ...
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'};
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub', ...
    'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For'};
    
num_train_per_cat = 50; % you can use smaller size of data set for debugging

fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels]=get_image_paths(data_path, categories, num_train_per_cat);  

%% Step 1: Represent each image with bag of words
fprintf('Using Bag of words representation for images\n') 
if ~exist('vocab.mat', 'file')
      fprintf('No existing visual word vocabulary found. Computing one from training images\n')
      
      num_words = 800; %Larger values will work better (to a point) but be slower to compute
      
      vocab = codebook(train_image_paths, num_words);
      save('vocab.mat', 'vocab');
      load('vocab.mat');
else 
      load('vocab.mat');
end       
train_image = bags_of_words(train_image_paths,vocab);
test_image  = bags_of_words(test_image_paths,vocab);

%% Step 2: Classify each test image by training and using the appropriate classifier
fprintf('Using SVM classifier to predict test set categories\n');

predicted_labels = svm(train_image, train_labels, test_image);

%% print out the result
create_results_webpage( train_image_paths,test_image_paths,train_labels,test_labels,categories,abbr_categories,predicted_labels);
   
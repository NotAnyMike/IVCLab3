function bag = bags_of_words(train_image_paths,vocab)

run ../vlfeat-0.9.21/toolbox/vl_setup

num = size(train_image_paths,1);
for i = 1:num
    in_image = imread(train_image_paths{i});
    in_image = single(in_image);
    [~, descriptor] = vl_dsift(in_image); % vl_sift(in_image);
    d = vl_alldist2(double(vocab), double(descriptor));
    d;
end
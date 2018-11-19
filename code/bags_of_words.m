function bag = bags_of_words(train_image_paths,vocab)

run ../vlfeat-0.9.21/toolbox/vl_setup

bag = zeros(size(train_image_paths,1),size(vocab,2));
step_p = 30;
binSize = 20;
num = size(train_image_paths,1);
for i = 1:num
    in_image = imread(train_image_paths{i});
    in_image = single(in_image);
    [~, descriptor] = vl_dsift(in_image,'Step',step_p,'size', binSize,'fast');%(in_image, 'fast'); % vl_sift(in_image);
    d = vl_alldist2(double(vocab), double(descriptor));
    [~,min_index] = min(d);
    h = hist(min_index,size(vocab,2));
    bag(i,:) = h;
end
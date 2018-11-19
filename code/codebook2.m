
function dictionary = codebook(image_paths,num_of_words)
fprintf('\nvocabulary\n');
num = size(image_paths,1);
container = [];
step_p = 30;
binSize = 20;
for i = 1:num
    img = single(imread(image_paths{i}));
    input_img = vl_imsmooth(img, 0.5);
    [~, sift_features] = vl_dsift(input_img,'Step',step_p,'size', binSize,'fast');
    %sift_features = vl_phow(img,'fast','true');
    container = [container;(single(sift_features'))];
    if mod(i,30) == 0
        fprintf('\n image %d \n',i);
    end
end


fprintf('\nstart to building vocaulary\n')
dictionary = vl_kmeans(container',num_of_words);
fprintf('\nfinish building vocaulary\n')


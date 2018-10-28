function image_new = tiny_images(image_paths)
fprintf('\nresize image\n');
num = size(image_paths,1);
new_size = 16; 
image_new = zeros(num,new_size*new_size);
for i = 1:num
    in_image = imread(image_paths{i});
    in_image = double(in_image);
    ns_image = imresize(in_image,[new_size, new_size]);
    image_new(i,:) = ns_image(:);
end

tiny_image = zeros(size(in_image));
tiny_image(1:new_size,1:new_size)=ns_image;

figure(1); clf; imagesc([in_image, tiny_image, imresize(ns_image,size(in_image),'nearest')]); title('Example image and its tiny representation');
%figure(1); clf; imagesc([in_image, ); title('Example image and its tiny representation');



function image_aligned = align_magnitude(img, ref)
img = img/max(img(:));
ref = ref/max(ref(:));
scale = [0.001:0.001:1];
snr = 10000;

for s=1:length(scale)
    tmp = norm(ref(:))/norm(ref(:)-(img(:)*scale(s)));
    temp(s) = tmp;
    if tmp<snr
        index = s;
    end
end
image_aligned = img * scale(index);

end
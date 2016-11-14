im1=imread('parede1.jpg');
im2=imread('parede2.jpg');
im1g=rgb2gray(im1);
im2g=rgb2gray(im2);
imagesc(im1g)
imagesc(im1g);colormap(gray);

convolution=conv2(double(im1g),ones(10),'same')
figure(2)
imagesc(convolution);colormap(gray);

filter=[-1 0 1;-1 0 1;-1 0 1];
im11=double(im1g);
gx=conv2(im11,filter,'same'); %derivative in x
figure(3)
imagesc(gx);colormap(gray); 
gy=conv2(im11,filter','same'); %derivative in y
figure(4)
imagesc(gy);colormap(gray); 

figure(5)
[cim1,r1,c1]=harris(im11,1,1000,3,1) %threshold=lambda1/lambda2

figure(6)
imagesc(im11); colormap(gray);
hold on
plot(c1,r1,'*');
im22=double(im2g);

figure(7)
[cim2,r2,c2]=harris(im22,1,1000,3,1) %threshold=lambda1/lambda2
figure(8)
imagesc(im22); colormap(gray);
hold on
plot(c2,r2,'*');
im22=double(im2);

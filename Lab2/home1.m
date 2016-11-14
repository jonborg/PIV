K=[525 0 319.5;
    0 525 239.5;
    0 0 1];

R12 =[0.2917   -0.5660    0.7711;
    0.4641    0.7886    0.4033;
   -0.8364    0.2402    0.4927];

T=[-2.1053 -0.7045 1.0838]';

depth1=imread('cardepth2.png');
im1=imread('car2.jpg');

depth2=imread('cardepth393.png');
im2=imread('car393.jpg');

figure(1)
imagesc(im1)
[u1 v1]=ginput(1);
figure(1);hold on;plot(u1,v1,'*r');hold off;
u1=round(u1);
v1=round(v1);
[u1 v1]

pw=zeros(3,1);
pw(3)= double(depth1(v1,u1))*0.001;
pw=inv(K)*[pw(3)*u1 pw(3)*v1 pw(3)]'


p=R12*pw+T
p=K*p;
u2=p(1)/p(3);
v2=p(2)/p(3);
[u2 v2]

figure(2)
imagesc(im2)
figure(2);hold on;plot(u2,v2,'*r');hold off;

% xyz1=get_xyzasus(depth1(:),[480 640],1:480*640,Depth_cam.K,1,0);
% xyz2=get_xyzasus(depth2(:),[480 640],1:480*640,Depth_cam.K,1,0);
% 
% rgbd1=get_rgbd(xyz1,im,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);
% rgbd2=get_rgbd(xyz2,im,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);
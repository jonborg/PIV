close all 
clear all

K=[525 0 319.5;

    0 525 239.5;

    0 0 1];

depth1=imread('cardepth2.png');
im1=imread('car2.jpg');
xyz1=get_xyzasus(depth1(:),[480 640],1:640*480,K,1,0);
cl1=reshape(im1,480*640,3);
m1=depth1>0;
imaux1=double(repmat(m1,[1,1,3])).*double(im1)/255;

depth2=imread('cardepth393.png');
im2=imread('car393.jpg');
xyz2=get_xyzasus(depth2(:),[480 640],1:640*480,K,1,0);
cl2=reshape(im2,480*640,3);
m2=depth2>0;
imaux2=double(repmat(m2,[1,1,3])).*double(im2)/255;

depth3=imread('depth641.png');
im3=imread('car641.jpg');
xyz3=get_xyzasus(depth2(:),[480 640],1:640*480,K,1,0);
cl3=reshape(im3,480*640,3);
m3=depth3>0;
imaux3=double(repmat(m3,[1,1,3])).*double(im3)/255;

% figure(1);
% imagesc(imaux1);
% [u1,v1]=ginput(5);
% 
% figure(2);
% imagesc(imaux2);
% [u2 v2]=ginput(5);
u1 =[  216.7007  324.5143 329.1022 352.0412 372.6864]';
v1 =[  53.7925 80.2550 111.8629 194.9257 78.7848]';
u2=[257.9910  224.7294  135.2670   42.3638  192.6147]';
v2=[87.6057  108.1876  124.3591  168.4632  130.9747]';

figure(3);
imagesc(imaux2);
[u3 v3]=ginput(5);

figure(4);
imagesc(imaux3);
[u4 v4]=ginput(5);

ind1=sub2ind([480 640],uint64(v1),uint64(u1));
ind2=sub2ind([480 640],uint64(v2),uint64(u2));
ind22=sub2ind([480 640],uint64(v3),uint64(u3));
ind3=sub2ind([480 640],uint64(v4),uint64(u4));

cent1=mean(xyz1(ind1,:))';
cent2=mean(xyz2(ind2,:))';
cent22=mean(xyz2(ind22,:))';
cent3=mean(xyz3(ind3,:))';

pc1=xyz1(ind1,:)'-repmat(cent1,1,5);
pc2=xyz2(ind2,:)'-repmat(cent2,1,5);
pc2=xyz2(ind22,:)'-repmat(cent22,1,5);
pc3=xyz3(ind3,:)'-repmat(cent3,1,5);


[a1 b1 c1]=svd(pc2*pc1')
R12=a1*c1'
[a2 b2 c2]=svd(pc3*pc2')
R23=a2*c2'

xyzt1=R12*(xyz1'-repmat(cent1,1,length(xyz1)));
xyzt2=xyz2'-repmat(cent2,1,length(xyz2));
xyzt22=R23*(xyz1'-repmat(cent22,1,length(xyz2)));
xyzt3=xyz3'-repmat(cent3,1,length(xyz3));
T2=cent3-R23*cent22;
T1=cent2-R12*cent1;

p1=pointCloud([xyzt1';xyzt2'],'Color',[cl1;cl2]);
p2=pointCloud([xyzt22';xyzt3'],'Color',[cl2;cl3]);
ptotal=pointCloud([xyzt1';xyzt2';xyzt3'],'Color',[cl1;cl2;cl3]);

figure(5);
showPointCloud(p1);

figure(6);
showPointCloud(p2);

figure(7);
showPointCloud(ptotal);
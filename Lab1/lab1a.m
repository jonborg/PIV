load calib_asus % Load Calibration Data for ASUS camera (similar to kinect)
load depth_10  % Load depth image
imagesc(depth_array)% Look at it...just a depth image
%Compute XYZ from depth image (u,v) and depth z(u,v)- CHECK FILE
xyz=get_xyz_asus(depth_array(:),[480 640],1:640*480,Depth_cam.K,1,0);
figure
%Display point cloud
p=pointCloud(xyz)
showPointCloud(p)
%Read RGB image
im=imread('rgb_image_10.png');
figure;
imagesc(im);
%Compute "virtual image" aligned with depth
rgbd=get_rgbd(xyz,im,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);
figure
imagesc([im; rgbd])
cl=reshape(rgbd,480*640,3);
p=pointCloud(xyz,'Color',cl);
figure
% Point cloud with colour per pixel
showPointCloud(p)
classdef pointCloud < matlab.mixin.Copyable & vision.internal.EnforceScalarHandle
% pointCloud Object for storing a 3-D point cloud.
%   ptCloud = pointCloud(xyzPoints) creates a point cloud object whose
%   coordinates are specified by an M-by-3 or M-by-N-by-3 matrix xyzPoints.
%
%   ptCloud = pointCloud(xyzPoints, Name, Value) specifies additional
%   name-value pair arguments described below:
%
%   'Color'             A uint8 matrix C specifying the color of each point
%                       in the point cloud. If the coordinates are an
%                       M-by-3 matrix, C must be an M-by-3 matrix, where
%                       each row contains a corresponding RGB value. If the
%                       coordinates are an M-by-N-by-3 matrix xyzPoints, C
%                       must be an M-by-N-by-3 matrix containing
%                       1-by-1-by-3 RGB values for each point.
%
%                       Default: []
%
%   'Normal'            A matrix NV specifying the normal vector at
%                       each point. If the coordinates are an M-by-3
%                       matrix, NV must be an M-by-3 matrix, where each row
%                       contains a corresponding normal vector. If the
%                       coordinates are an M-by-N-by-3 matrix xyzPoints, NV
%                       must be an M-by-N-by-3 matrix containing
%                       1-by-1-by-3 normal vector for each point.
%
%                       Default: []
%
%   pointCloud properties :
%   Location        - Matrix of [X, Y, Z] point coordinates (read only)
%   Color           - Matrix of point RGB colors
%   Normal          - Matrix of [NX, NY, NZ] point normal directions 
%   Count           - Number of points (read only)
%   XLimits         - The range of coordinates along X-axis (read only)
%   YLimits         - The range of coordinates along Y-axis (read only)
%   ZLimits         - The range of coordinates along Z-axis (read only)
%
%   pointCloud methods:
%   findNearestNeighbors   - Retrieve K nearest neighbors of a point
%   findNeighborsInRadius  - Retrieve neighbors of a point within a specified radius
%   findPointsInROI        - Retrieve points within a region of interest
%   select                 - Select points specified by index
%   removeInvalidPoints    - Remove invalid points
%
%   Notes 
%   ----- 
%   - To reduce memory use, point locations are suggested to be stored as single.
%
%   Example : Find the shortest distance between two point clouds
%   -------
%   ptCloud1 = pointCloud(rand(100, 3, 'single'));
%        
%   ptCloud2 = pointCloud(1+rand(100, 3, 'single'));
%
%   minDist = inf;
%
%   % Find the nearest neighbor in ptCloud2 for each point in ptCloud1
%   for i = 1 : ptCloud1.Count
%       point = ptCloud1.Location(i,:);
%       [~, dist] = findNearestNeighbors(ptCloud2, point, 1);
%       if dist < minDist 
%           minDist = dist;
%       end
%   end
%
%   See also showPointCloud, depthToPointCloud, reconstructScene, pcread,
%   pcwrite, pctransform, pcdenoise, pcdownsample, pcregrigid, pcmerge
 
%  Copyright 2014 The MathWorks, Inc.
%
% References
% ----------
% Marius Muja and David G. Lowe, "Fast Approximate Nearest Neighbors with
% Automatic Algorithm Configuration", in International Conference on
% Computer Vision Theory and Applications (VISAPP'09), 2009

    properties (GetAccess = public, SetAccess = private)
        % Location is an M-by-3 or M-by-N-by-3 matrix. Each entry specifies
        % the x, y, z coordinates of a point.
        Location = single([]);
    end
    
    properties (Access = public)
        % Color is an M-by-3 or M-by-N-by-3 uint8 matrix. Each entry
        % specifies the RGB color of a point.
        Color = uint8([]);
        
        % Normal is an M-by-3 or M-by-N-by-3 matrix. Each entry
        % specifies the x, y, z component of a normal vector.
        Normal = single([]);                
    end
    
    properties(Access = protected, Transient)
        Kdtree = [];
    end
       
    properties(Dependent)
        Count;
        XLimits;
        YLimits;
        ZLimits;
    end
    
    methods
        %==================================================================
        % Constructor
        %==================================================================
        % pcd = pointCloud(xyzPoints);
        % pcd = pointCloud(..., 'Color', C);
        % pcd = pointCloud(..., 'Normal', nv);
        function this = pointCloud(varargin)
            narginchk(1, 5);
            
            [xyzPoints, C, nv] = validateAndParseInputs(varargin{:}); 
            
            this.Location = xyzPoints;
            this.Color = C;
            this.Normal = nv;
        end
        
        %==================================================================
        % K nearest neighbor search
        %==================================================================
        function [indices, dists] = findNearestNeighbors(this, point, K, varargin)
            %findNearestNeighbors Find the nearest neighbors of a point.
            %
            %  [indices, dists] = findNearestNeighbors(ptCloud, point, K)
            %  returns the K nearest neighbors of a query point. The point
            %  is an [X, Y, Z] vector. indices is a column vector that 
            %  contains K linear indices to the stored points in the point 
            %  cloud. dists is a column vector that contains K Euclidean 
            %  distances to the point. 
            %
            %  [...] = findNearestNeighbors(... , Name, Value)
            %  specifies additional name-value pairs described below:
            %  
            %  'Sort'          True or false. When set to true,
            %                  the returned indices are sorted in the
            %                  ascending order based on the distance
            %                  from a query point.
            %
            %                  Default: false
            %
            %  'MaxLeafChecks' An integer specifying the number of leaf
            %                  nodes that are searched in Kdtree. If the tree
            %                  is searched partially, the returned result may
            %                  not be exact. A larger number may increase the
            %                  search accuracy but reduce efficiency. When this
            %                  number is set to inf, the whole tree will be
            %                  searched to return exact search result. 
            %
            %                  Default: inf
            %
            %  Example : Find K-nearest neighbors
            %  ---------
            %  % Create a point cloud object with randomly generated points
            %  ptCloud = pointCloud(1000*rand(100,3,'single'));
            %
            %  % Define a query point and number of neighbors
            %  point = [50, 50, 50];
            %  K = 10;
            %
            %  % Get the indices and distances of 10 nearest points
            %  [indices, dists] = findNearestNeighbors(ptCloud, point, K);

            narginchk(3, 7);

            validateattributes(point, {'single', 'double'}, ...
                {'real', 'nonsparse', 'finite', 'size', [1, 3]}, mfilename, 'point');
            
            if isa(this.Location, 'single')
                point = single(point);
            else
                point = double(point);
            end

            validateattributes(K, {'single', 'double'}, ...
                {'nonsparse', 'scalar', 'positive', 'integer'}, mfilename, 'K');

            K = min(double(K), numel(this.Location)/3);

            nargoutchk(0, 3);

            [doSort, MaxLeafChecks] = validateAndParseSearchOption(varargin{:});

            % Use bruteforce if there are fewer than 500 points
            if numel(this.Location)/3 < 500
                if ismatrix(this.Location)
                    allDists = visionSSDMetric(point', this.Location');
                else
                    allDists = visionSSDMetric(point', reshape(this.Location, [], 3)');
                end
                % This function will ensure returning actual number of neighbors
                % The result is already sorted
                [dists, indices] = vision.internal.partialSort(allDists, K);
                tf = isfinite(dists);
                indices = indices(tf);
                dists = dists(tf);
            else
                this.buildKdtree();

                searchOpts.checks = MaxLeafChecks;
                searchOpts.eps = 0;
                [indices, dists, valid] = this.Kdtree.knnSearch(point, K, searchOpts);
                
                % This step will ensure returning actual number of neighbors
                indices = indices(1:valid);
                dists = dists(1:valid);

                % Sort the result if specified
                if doSort
                    [dists, IND] = sort(dists);
                    indices = indices(IND);
                end
            end

            if nargout > 1
                dists = sqrt(dists);
            end
        end

        %==================================================================
        % Radius search
        %==================================================================
        function [indices, dists] = findNeighborsInRadius(this, point, radius, varargin)
            %findNeighborsInRadius Find the neighbors within a radius.
            %
            %  [indices, dists] = findNeighborsInRadius(ptCloud, point, radius) 
            %  returns the neighbors within a radius of a query point.
            %  The query point is an [X, Y, Z] vector. indices is a column
            %  vector that contains the linear indices to the stored points in
            %  the point cloud object ptCloud. dists is a column vector that
            %  contains the Euclidean distances to the query point.
            %
            %  [...] = findNeighborsInRadius(... , Name, Value) specifies
            %  specifies additional name-value pairs described below:
            %  
            %  'Sort'          True or false. When set to true,
            %                  the returned indices are sorted in the
            %                  ascending order based on the distance
            %                  from a query point.
            %
            %                  Default: false
            %
            %  'MaxLeafChecks' An integer specifying the number of leaf
            %                  nodes that are searched in Kdtree. If the tree
            %                  is searched partially, the returned result may
            %                  not be exact. A larger number may increase the
            %                  search accuracy but reduce efficiency. When this
            %                  number is set to inf, the whole tree will be
            %                  searched to return exact search result. 
            %
            %                  Default: inf
            %
            %   Example : Find neighbors within a given radius using Kdtree
            %   -----------------------------------------------------------
            %   % Create a point cloud object with randomly generated points
            %   ptCloud = pointCloud(100*rand(1000, 3, 'single'));
            %
            %   % Define a query point and search radius
            %   point = [50, 50, 50];
            %   radius =  5;
            %
            %   % Get all the points within a radius
            %   [indices, dists] = findNeighborsInRadius(ptCloud, point, radius);
        
            narginchk(3, 7);

            validateattributes(point, {'single', 'double'}, ...
                {'real', 'nonsparse', 'finite', 'size', [1, 3]}, mfilename, 'point');

            if isa(this.Location, 'single')
                point = single(point);
            else
                point = double(point);
            end

            validateattributes(radius, {'single', 'double'}, ...
                {'nonsparse', 'scalar', 'nonnegative', 'finite'}, mfilename, 'radius');
            
            radius = double(radius);
            
            nargoutchk(0, 3);
            
            [doSort, MaxLeafChecks] = validateAndParseSearchOption(varargin{:});
            
            % Use bruteforce if there are less than 500 points
            if numel(this.Location)/3 < 500
                if ismatrix(this.Location)
                    allDists = visionSSDMetric(point', this.Location');
                else
                    allDists = visionSSDMetric(point', reshape(this.Location, [], 3)');
                end
                indices = uint32(find(allDists <= radius^2))';
                dists = allDists(indices)';
                tf = isfinite(dists);
                indices = indices(tf);
                dists = dists(tf);
            else
                this.buildKdtree();

                searchOpts.checks = MaxLeafChecks;
                searchOpts.eps = 0;
                [indices, dists] = this.Kdtree.radiusSearch(point, radius, searchOpts);
            end
            
            % Sort the result if specified
            if doSort
                [dists, IND] = sort(dists);
                indices = indices(IND);
            end
            
            if nargout > 1
                dists = sqrt(dists);
            end
        end       
        
        %==================================================================
        % Box search
        %==================================================================
        function indices = findPointsInROI(this, roi)
            %findPointsInROI Find points within a region of interest.
            %
            %  indices = findPointsInROI(ptCloud, roi) returns the points
            %  within a region of interest, roi. The roi is a cuboid
            %  specified as a 3-by-2 matrix in the format of [xmin,
            %  xmax; ymin, ymax; zmin, zmax]. indices is a column vector
            %  that contains the linear indices to the stored points in the
            %  point cloud object, ptCloud.
            %
            %   Example : Find points within a given cuboid
            %   -------------------------------------------
            %   % Create a point cloud object with randomly generated points
            %   ptCloudA = pointCloud(100*rand(1000, 3, 'single'));
            %
            %   % Define a cuboid
            %   roi = [0, 50; 0, inf; 0, inf];
            %
            %   % Get all the points within the cuboid
            %   indices = findPointsInROI(ptCloudA, roi);
            %   ptCloudB = select(ptCloudA, indices);
            %
            %   showPointCloud(ptCloudA.Location, 'r');
            %   hold on;
            %   showPointCloud(ptCloudB.Location, 'g');
            %   hold off;
            
            narginchk(2, 2);

            validateattributes(roi, {'single', 'double'}, ...
                {'real', 'nonsparse', 'size', [3, 2]}, mfilename, 'roi');
            
            if any(roi(:,1)>roi(:,2))
                error(message('vision:pointcloud:invalidROI'));
            end

            roi = double(roi);
            
            nargoutchk(0, 2);
            
            % Use bruteforce if there are less than 500 points
            if numel(this.Location)/3 < 500
                if ismatrix(this.Location)
                    tf = this.Location(:,1)>=roi(1)&this.Location(:,1)<=roi(4) ...
                        &this.Location(:,2)>=roi(2)&this.Location(:,2)<=roi(5) ...
                        &this.Location(:,3)>=roi(3)&this.Location(:,3)<=roi(6);
                    indices = uint32(find(tf));
                else
                    tf = this.Location(:,:,1)>=roi(1)&this.Location(:,:,1)<=roi(4) ...
                        &this.Location(:,:,2)>=roi(2)&this.Location(:,:,2)<=roi(5) ...
                        &this.Location(:,:,3)>=roi(3)&this.Location(:,:,3)<=roi(6);
                    indices = uint32(find(tf));
                end
            else
                this.buildKdtree();

                indices = this.Kdtree.boxSearch(roi);
            end                        
        end
        
        %==================================================================
        % Obtain a subset of this point cloud object
        %==================================================================
        function ptCloudOut = select(this, varargin)
            %select Select points specified by index.
            %
            %  ptCloudOut = select(ptCloud, indices) returns a pointCloud
            %  object that contains the points selected using linear
            %  indices.
            %
            %  ptCloudOut = select(ptCloud, row, column) returns a pointCloud
            %  object that contains the points selected using row and
            %  column subscripts. This syntax applies only to organized
            %  point cloud (M-by-N-by-3).
            %
            %  Example : Downsample a point cloud with fixed step
            %  -------------------------------------------------
            %   ptCloud = pcread('teapot.ply');
            %
            %   % Downsample a point cloud with fixed step size 
            %   stepSize = 10;
            %   indices = 1:stepSize:ptCloud.Count;
            %
            %   ptCloudOut = select(ptCloud, indices);
            %
            %   showPointCloud(ptCloudOut);

            narginchk(2, 3);
            
            if nargin == 2
                indices = varargin{1};
                validateattributes(indices, {'numeric'}, ...
                    {'real','nonsparse', 'vector', 'integer'}, mfilename, 'indices');
            else
                % Subscript syntax is only for organized point cloud
                if ndims(this.Location) ~= 3
                    error(message('vision:pointcloud:organizedPtCloudOnly'));
                end
                
                row = varargin{1};
                column = varargin{2};
                validateattributes(row, {'numeric'}, ...
                    {'real','nonsparse', 'vector', 'integer'}, mfilename, 'row');
                validateattributes(column, {'numeric'}, ...
                    {'real','nonsparse', 'vector', 'integer'}, mfilename, 'column');
                indices = sub2ind([size(this.Location,1), size(this.Location,2)], row, column);
            end
            
            % Obtain the subset for every property
            [loc, c, nv] = this.subsetImpl(indices);
            
            ptCloudOut = pointCloud(loc, 'Color', c, 'Normal', nv);
        end
        
        %==================================================================
        % Remove invalid points from this point cloud object
        %==================================================================
        function ptCloudOut = removeInvalidPoints(this)
            %removeInvalidPoints Remove invalid points.
            %
            %  ptCloudOut = removeInvalidPoints(ptCloud) removes points
            %  whose coordinates have Inf or NaN.
            %
            %  Note :
            %  ---------
            %  An organized point cloud (M-by-N-by-3) will become
            %  unorganized (X-by-3) after calling this function.
            %   
            %  Example : Remove NaN valued points from a point cloud
            %  ---------------------------------------------------------
            %  ptCloud = pointCloud(nan(100,3))
            %
            %  ptCloud = removeInvalidPoints(ptCloud)

            % Find all valid points
            tf = isfinite(this.Location);
            if ismatrix(this.Location)
                indices = (sum(tf, 2) == 3);
            else
                indices = (sum(reshape(tf, [], 3), 2) == 3);
            end

            [loc, c, nv] = this.subsetImpl(indices);
            ptCloudOut = pointCloud(loc, 'Color', c, 'Normal', nv);
        end

    end
    
    methods
        %==================================================================
        % Writable Property
        %==================================================================
        function set.Color(this, value) 
            validateattributes(value,{'uint8'}, {'real','nonsparse'});
            if ~isempty(value) && ~isequal(size(value), size(this.Location)) %#ok<MCSUP>
                error(message('vision:pointcloud:unmatchedXYZColor'));
            end
            this.Color = value;
        end
        
        function set.Normal(this, value)
            validateattributes(value,{'single', 'double'}, {'real','nonsparse'});
            if ~isempty(value) && ~isequal(size(value), size(this.Location)) %#ok<MCSUP>
                error(message('vision:pointcloud:unmatchedXYZNormal'));
            end
            if isa(this.Location,'double') %#ok<MCSUP>
                value = double(value);
            else
                value = single(value);
            end
            this.Normal = value;
        end
        
        %==================================================================
        % Dependent Property
        %==================================================================
        function xlim = get.XLimits(this)
            tf = ~isnan(this.Location);
            if ismatrix(this.Location)
                tf = (sum(tf, 2)==3);
                xlim = [min(this.Location(tf, 1)), max(this.Location(tf, 1))];
            else
                tf = (sum(tf, 3)==3);
                X = this.Location(:, :, 1);
                xlim = [min(X(tf)), max(X(tf))];
            end                
        end  
        %==================================================================
        function ylim = get.YLimits(this)
            tf = ~isnan(this.Location);
            if ismatrix(this.Location)
                tf = (sum(tf, 2)==3);
                ylim = [min(this.Location(tf, 2)), max(this.Location(tf, 2))];
            else
                tf = (sum(tf, 3)==3);
                Y = this.Location(:, :, 2);
                ylim = [min(Y(tf)), max(Y(tf))];
            end                
        end  
        %==================================================================
        function zlim = get.ZLimits(this)
            tf = ~isnan(this.Location);
            if ismatrix(this.Location)
                tf = (sum(tf, 2)==3);
                zlim = [min(this.Location(tf, 3)), max(this.Location(tf, 3))];
            else
                tf = (sum(tf, 3)==3);
                Z = this.Location(:, :, 3);
                zlim = [min(Z(tf)), max(Z(tf))];
            end                
        end
        %==================================================================
        function count = get.Count(this)
            if ismatrix(this.Location)
                count = size(this.Location, 1);
            else
                count = size(this.Location, 1)*size(this.Location, 2);
            end                
        end  
    end
    
    methods (Access = public, Hidden)
        %==================================================================
        % helper function to get subset for each property
        %==================================================================
        function [loc, c, nv] = subsetImpl(this, indices)
            if ~isempty(this.Location)
                if ismatrix(this.Location)
                    loc = this.Location(indices, :);
                else
                    loc = reshape(this.Location, [], 3);
                    loc = loc(indices, :);
                end
            else
                loc = zeros(0, 3, 'single');
            end
            
            if ~isempty(this.Color)
                if ismatrix(this.Color)
                    c = this.Color(indices, :);
                else
                    c = reshape(this.Color, [], 3);
                    c = c(indices, :);
                end
            else
                c = uint8([]);
            end
            
            if ~isempty(this.Normal)
                if ismatrix(this.Normal)
                    nv = this.Normal(indices, :);
                else
                    nv = reshape(this.Normal, [], 3);
                    nv = nv(indices, :);
                end
            else
                 if isa(loc, 'single')
                    nv = single.empty;
                 else
                    nv = double.empty;
                 end
            end
        end
        
        %==================================================================
        % helper function to support multiple queries in KNN search
        % indices, dists: K-by-numQueries
        % valid: 1-by-K
        % Note, the algorithm may return less than K results for each
        % query. Therefore, only 1:valid(n) in n-th column of indices and
        % dists are valid results. Invalid indices are all zeros.
        %==================================================================
        function [indices, dists, valid] = multiQueryKNNSearchImpl(this, points, K)
            % Validate the inputs
            validateattributes(points, {'single', 'double'}, ...
                {'real', 'nonsparse', 'size', [NaN, 3]}, mfilename, 'points');

            if isa(this.Location, 'single')
                points = single(points);
            else
                points = double(points);
            end

            validateattributes(K, {'single', 'double'}, ...
                {'nonsparse', 'scalar', 'positive', 'integer'}, mfilename, 'K');

            K = min(double(K), numel(this.Location)/3);
            
            this.buildKdtree();

            % Use exact search in Kdtree
            searchOpts.checks = 0;
            searchOpts.eps = 0;
            [indices, dists, valid] = this.Kdtree.knnSearch(points, K, searchOpts);
        end
    end

    methods (Access = protected)        
        %==================================================================
        % helper function to index data
        %==================================================================
        function buildKdtree(this)
            if isempty(this.Kdtree)
                % Build a Kdtree to index the data
                this.Kdtree = vision.internal.Kdtree();
                this.Kdtree.index(this.Location);                                        
            else
                % Ensure that we keep indexing the same dataset.
                this.Kdtree.checkIndex(this.Location); 
            end            
        end
    end    
    
    methods(Static, Access=private)
        %==================================================================
        % load object 
        %==================================================================
        function this = loadobj(that)
            this = pointCloud(that.Location,...
                'Color', that.Color,...
                'Normal', that.Normal);
        end  
    end
    
    methods(Access=private)
        %==================================================================
        % save object 
        %==================================================================
        function s = saveobj(this)
            % save properties into struct
            s.Location      = this.Location;
            s.Color         = this.Color;
            s.Normal        = this.Normal;
        end
    end
    
    methods(Access=protected)
        %==================================================================
        % copy object 
        %==================================================================
        % Override copyElement method:
        function cpObj = copyElement(obj)
            % Make a copy except the internal Kdtree
            cpObj = pointCloud(obj.Location, 'Color', obj.Color, 'Normal', obj.Normal);
        end
    end    
end

%==================================================================
% parameter validation
%==================================================================
function [xyzPoints, C, nv] = validateAndParseInputs(varargin)
    % Validate and parse inputs
    narginchk(1, 5);

    parser = inputParser;
    parser.CaseSensitive = false;
    % Parse the arguments according to the format of the first argument

    if ismatrix(varargin{1})
        dims = [NaN, 3];
    else
        dims = [NaN, NaN, 3];
    end

    parser.addRequired('xyzPoints', @(x)validateattributes(x,{'single', 'double'}, {'real','nonsparse','size', dims}));            
    parser.addParameter('Color', uint8([]), @(x)validateattributes(x,{'uint8'}, {'real','nonsparse'}));
    parser.addParameter('Normal', single([]),  @(x)validateattributes(x,{'single', 'double'}, {'real','nonsparse'}));

    parser.parse(varargin{:});

    xyzPoints = parser.Results.xyzPoints;
    C = parser.Results.Color;
    nv = parser.Results.Normal;           
    if isa(xyzPoints, 'single')
        nv = single(nv);
    else
        nv = double(nv);
    end
end                                
%==================================================================
% parameter validation for search
%==================================================================
function [doSort, maxLeafChecks] = validateAndParseSearchOption(varargin)
persistent p;
if isempty(p)
    % Validate and parse search options
    p = inputParser;
    p.CaseSensitive = false;
    p.addParameter('Sort', false, @(x)validateattributes(x, {'logical'}, {'scalar'}));
    p.addParameter('MaxLeafChecks', inf, @validateMaxLeafChecks);
    parser = p;
else
    parser = p;
end

parser.parse(varargin{:});

doSort = parser.Results.Sort;
maxLeafChecks = parser.Results.MaxLeafChecks;
if isinf(maxLeafChecks)
    % 0 indicates infinite search in internal function
    maxLeafChecks = 0;
end

end
%==================================================================
function validateMaxLeafChecks(value)
    % Validate MaxLeafChecks
    if any(isinf(value))
        validateattributes(value,{'double'}, {'real','nonsparse','scalar','positive'});
    else
        validateattributes(value,{'double'}, {'real','nonsparse','scalar','integer','positive'});
    end
end

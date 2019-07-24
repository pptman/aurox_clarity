import numpy as np
import cv2

class clarity_processor:
    width = 0
    height = 0
    defX = cv2.UMat()
    defY = cv2.UMat()

    def __init__(self,cal_img):
        self.width = int(np.size(cal_img,1)/2)
        self.height = np.size(cal_img,0)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 120
        params.thresholdStep = 5

        params.filterByColor = True
        params.blobColor = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 40

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.maxCircularity = 1.0

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.maxConvexity = 1.0

        detector = cv2.SimpleBlobDetector_create(params)

        normVal = 255.0/np.max(cal_img)
        im = np.uint8(cal_img*normVal)

        # find spots in left image
        keypoints_l = detector.detect(im[:,0:self.width])
        nl = len(keypoints_l)
        posl = np.asarray([ keypoints_l[i].pt for i in range(nl) ])
        d=5             # reject points found within d pixels of image edge
        posl = posl[(posl[:,0]>d)&(posl[:,0]<(self.width-d))&(posl[:,1]>d)&(posl[:,1]<(self.height-d)),:]
        nl = len(posl)

        # find spots in right image
        keypoints_r = detector.detect(im[:,self.width:])
        nr = len(keypoints_r)

        posr = np.asarray([ keypoints_r[i].pt for i in range(nr) ])
        posr = posr[(posr[:,0]>d)&(posr[:,0]<(self.width-d))&(posr[:,1]>d)&(posr[:,1]<(self.height-d)),:]
        nr=len(posr)

        # find spots' nearest neighbours left
        distml=np.zeros((nl,nl))
        for i in range(nl):
            distml[i,:] = np.hypot(posl[:,0]-posl[i,0],posl[:,1]-posl[i,1])
        dlsort=np.sort(distml.ravel())
        dmidl=dlsort[int(3.5*nl)]
        dlowl=0.91*dmidl
        dhil=1.09*dmidl
        distnuml=np.sum((distml>dlowl)&(distml<dhil),0)

        # find spots' nearest neighbours right
        distmr=np.zeros((nr,nr))
        for i in range(nr):
            distmr[i,:] = np.hypot(posr[:,0]-posr[i,0],posr[:,1]-posr[i,1])
        drsort=np.sort(distmr.ravel())
        dmidr=drsort[int(3.5*nr)]
        dlowr=0.91*dmidr
        dhir=1.09*dmidr
        distnumr=np.sum((distmr>dlowr)&(distmr<dhir),0)

        # find left 6-way co-ordinated spots
        pos6l=posl[distnuml==6,:]  # positions on left that are 6-way co-ordinated
        indxl=np.arange(nl)
        diffl=np.zeros((nl,2))
        for i in indxl[distnuml==6] :                           # iterate over 6-way co-ordinated points
            distindx=(distml[i,:]>dlowl)&(distml[i,:]<dhil)     # the points that are required distance from current point
            diffl[i,:]=np.sum(posl[distindx,:]-posl[i,:],0)     # find centroid of the 6 co-ordinated points relative to current

        diff6l=diffl[indxl[distnuml==6],:]                      # a list of just the centroids of 6 co-ordinated points
        n6l = len(pos6l)                                        # number of 6-way coordinated points

        # find right 6-way co-ordinated spots
        pos6r=posr[distnumr==6,:]  # positions on right that are 6-way co-ordinated
        indxr=np.arange(nr)
        diffr=np.zeros((nr,2))
        for i in indxr[distnumr==6] :                           # iterate over 6-way co-ordinated points
            distindx=(distmr[i,:]>dlowr)&(distmr[i,:]<dhir)     # the points that are required distance from current point
            diffr[i,:]=np.sum(posr[distindx,:]-posr[i,:],0)     # find centroid of the 6 co-ordinated points relative to current

        diff6r=diffr[indxr[distnumr==6],:]                      # a list of just the centroids of 6 co-ordinated points
        n6r = len(pos6r)                                        # number of 6-way coordinated points

        # find matched 6-way co-ordinated spot pairs
        pmatch6l = []
        pmatch6r = []

        for i in range(n6l) :                                   # loop over all 6-way co-ordinated left spots
             dist6=np.hypot(pos6r[:,1]-pos6l[i,1],self.width-1-pos6r[:,0]-pos6l[i,0]); # the distance of all right spots from this left spot
             distoffs=np.hypot(diff6l[i,1]-diff6r[:,1],diff6l[i,0]+diff6r[:,0]);   # the distance between the right and left 6-spot centroids
             for j in range(n6r) :                              # loop over all 6-way co-ordinated spots on right
                 if (dist6[j]<4*dmidl)&(distoffs[j]<dmidl/4) :  # select spots as pairs if they are close enough and if the centroids are close enough
                     pmatch6l=np.append(pmatch6l,pos6l[i,:]);   # add to list of left and right matched spots
                     pmatch6r=np.append(pmatch6r,pos6r[j,:]);
        nmatch6=int(len(pmatch6l)/2)
        pmatch6l=np.reshape(pmatch6l,(nmatch6,2))
        pmatch6r=np.reshape(pmatch6r,(nmatch6,2))

        # find affine transform based on 6-way matches
        [retval, map6]=cv2.solve(np.c_[pmatch6r,np.ones(nmatch6)],pmatch6l,flags=cv2.DECOMP_SVD)

        # transform all right spot positions to find matches on left
        posrt=np.matmul(np.c_[posr,np.ones(nr)],map6)

        pmatchl=[]
        pmatchr=[]
        for i in range(nl):
            dist = np.hypot(posl[i,1]-posrt[:,1],posl[i,0]-posrt[:,0])
            mind = min(dist)
            if (mind<5) :
                pmatchl=np.append(pmatchl,posl[i,:])
                pmatchr=np.append(pmatchr,posr[np.argmin(dist),:])

        nmatch = int(len(pmatchl)/2)
        pmatchl=np.reshape(pmatchl,(nmatch,2))
        pmatchr=np.reshape(pmatchr,(nmatch,2))

        # find polynomial transform for all matched pairs
        # 1, x, y, x^2, y^2, x*y, x^2*y, x*y^2, x^3, y^3
        rmat=np.c_[np.ones(nmatch),pmatchl,pmatchl*pmatchl, pmatchl[:,0]*pmatchl[:,1],
                   pmatchl[:,0]*pmatchl[:,0]*pmatchl[:,1], pmatchl[:,0]*pmatchl[:,1]*pmatchl[:,1], pmatchl*pmatchl*pmatchl]
        [retval, map]=cv2.solve(rmat,pmatchr,flags=cv2.DECOMP_SVD)

        # find deformation maps

        [x, y] = np.meshgrid(np.arange(self.width),np.arange(self.height))

        self.defXcpu = np.float32(map[0,0]+map[1,0]*x+map[2,0]*y+map[3,0]*x*x+map[4,0]*y*y+map[5,0]*x*y+map[6,0]*x*x*y+map[7,0]*x*y*y+map[8,0]*x*x*x+map[9,0]*y*y*y)
        self.defYcpu = np.float32(map[0,1]+map[1,1]*x+map[2,1]*y+map[3,1]*x*x+map[4,1]*y*y+map[5,1]*x*y+map[6,1]*x*x*y+map[7,1]*x*y*y+map[8,1]*x*x*x+map[9,1]*y*y*y)

        self.defX = cv2.UMat(self.defXcpu)
        self.defY = cv2.UMat(self.defYcpu)

    def process(self, img, sub_factor=1.0):
        # Takes 1 combined numpy (Mat) array and converts to UMat on the fly from numpy subimages
        # OPENCV transparent interface will use OPENCL for processing
        # Approx 9.2 ms processing time on macbook pro (i7 + Intel Iris Pro)
        result = cv2.subtract(cv2.UMat(img[:,0:self.width]),
                              cv2.remap(cv2.multiply(cv2.UMat(img[:,self.width:]),sub_factor),self.defX,self.defY,cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_CONSTANT,borderValue=0))
        return result

    def process_gpu1(self, img_l, img_r, sub_factor=1.0):
        # Takes 2 UMat images as arguments,
        # OPENCV transparent interface will use OPENCL for processing
        # Approx 2.8 ms processing time on macbook pro (i7 + Intel Iris Pro), but your images need to be separate UMats already
        # result = cv2.subtract(img_l,
        #                       cv2.remap(cv2.multiply(img_r,sub_factor),self.defX,self.defY,cv2.INTER_LINEAR,
        #                                              borderMode=cv2.BORDER_CONSTANT,borderValue=0))
        result = cv2.scaleAdd(
            cv2.remap(img_r, self.defX, self.defY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0),
            -sub_factor, img_l)
        return result

    def process_gpu2(self, img, sub_factor=1.0):
        # Takes 1 combined numpy (Mat) array and converts to UMat on the fly
        # Converts combined image to UMat on gpu then transparent interface produces 2 sub image UMats in OPENCL
        # OPENCV transparent interface will use OPENCL for processing
        # Approx 3.2 ms processing time on macbook pro (i7 + Intel Iris Pro)
        uimg = cv2.UMat(img)
        img_l = cv2.UMat(uimg, (0, self.height), (0, self.width))
        img_r = cv2.UMat(uimg, (0, self.height), (self.width, 2*self.width))
        result = cv2.subtract(img_l,
                              cv2.remap(cv2.multiply(img_r,sub_factor),self.defX,self.defY,cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_CONSTANT,borderValue=0))
        return result

    def process_gpu3(self, img, sub_factor=1.0):
        # Takes 1 combined numpy (Mat) array and converts to UMat on the fly
        # Converts combined image to UMat on gpu then transparent interface produces 2 sub image UMats in OPENCL
        # OPENCV transparent interface will use OPENCL for processing, uses scaleAdd routine for subtraction step
        # Approx 2.7 ms processing time on macbook pro (i7 + Intel Iris Pro)
        uimg = cv2.UMat(img)
        img_l = cv2.UMat(uimg, (0, self.height), (0, self.width))
        img_r = cv2.UMat(uimg, (0, self.height), (self.width, 2*self.width))
        result = cv2.scaleAdd(cv2.remap(img_r,self.defX,self.defY,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0),
                              -sub_factor,img_l)
        return result

    def process_cpu(self, img, sub_factor=1.0):
        # Takes 1 combined numpy (Mat) array and converts to UMat on the fly from numpy subimages
        # OPENCV transparent interface will use OPENCL for processing
        # Approx 7.2 ms processing time on macbook pro (i7 + Intel Iris Pro)
        result = cv2.subtract(img[:,0:self.width],
                              cv2.remap(cv2.multiply(img[:,self.width:],sub_factor),self.defXcpu,self.defYcpu,cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_CONSTANT,borderValue=0))
        return result

    def process_cpu1(self, img, sub_factor=1.0):
        # Takes 1 combined numpy (Mat) array and converts to UMat on the fly from numpy subimages
        # OPENCV transparent interface will use OPENCL for processing
        # Approx 7.2 ms processing time on macbook pro (i7 + Intel Iris Pro)
        result = cv2.scaleAdd(
            cv2.remap(img[:,self.width:], self.defXcpu, self.defYcpu, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0),
            -sub_factor, img[:,0:self.width])
        return result


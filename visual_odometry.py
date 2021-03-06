import cv2
import numpy as np
import matplotlib.pyplot as plt
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import os
import random



frames=[]
path='stereo/centre/'
for frame in os.listdir(path):
    frames.append(frame)
    frames.sort()

fx, fy, cx, cy, G_camera_image, LUT= ReadCameraModel('model/')


K = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]]) # Camera Calibration Matrix of the model
# takes corners of two images and return fundamental metrix
#when we have features more than 8 we will stack them in A matrix and use svd , Ax=0


def fundamentalMatrix(feature1, feature2): 
    A_x = np.empty((8, 9))

    for i in range(0, len(feature1)):
        x1 = feature1[i][0]
        y1 = feature1[i][1]
        x2 = feature2[i][0]
        y2 = feature2[i][1]
        A_x[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    u, s, v = np.linalg.svd(A_x, full_matrices=True)  # Taking SVD of the matrix
    f = v[-1].reshape(3,3) # we have 8 equations for 9 unknowns. Thus, the last column of V is the true solution
    u1,s1,v1 = np.linalg.svd(f) 
    s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) #due to noise in the correspondences, 
    #the estimated F matrix can be of rank 3.So, to enfore the rank 2 constraint, the last singular value of the estimated F must be set to zero.
    F = u1 * s2 *v1.T 
    return F  #7 Dof (defined in original image space)

def Homogenousmatrix(R, T):
    z = np.column_stack((R, T))
    a = np.array([0, 0, 0, 1])
    z = np.vstack((z, a))
    return z




# generates eight random points in frame 1 and 2

def get_eights_points(features1, features2):
    
    # we need eight random points

    finalFundMatrix = np.zeros((3,3))

   
    # RANSAC Algorithm
    for i in range(0, 50): 
        count = 0
        eightpoints = [] 
        eight1 = [] # eight points for frame 1 
        eight2 = [] # eight points for frame 2
        tempfeature1 = [] 
        tempfeature2 = []
        
        while(True): 
            num = random.randint(0, len(features1)-1)
            if num not in eightpoints:
                eightpoints.append(num)
            if len(eightpoints) == 8:
                break

        for i in eightpoints: 
            eight1.append([features1[i][0], features1[i][1]]) 
            eight2.append([features2[i][0], features2[i][1]])

    return eight1,eight2

    #computes x2.T * F * x1


def inlier_threshold(x1,x2,F):

    x1_=np.array([x1[0],x1[1],1]).T
    x2_=np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x2_,F)),x1_)))


def get_inliers(features1,features2,Fundamental_Matrix):
     # If x2.T * F * x1 is less than threshold (0.01) then it is considered as Inlier
    inlier1 = [] # inliers features from frame1
    inlier2 = [] # inliers features from the frame2
    count=0
    Inliers = 0
    finalFund_Matrix = np.zeros((3,3))
    temp1=[]
    temp2=[]
    for i in range(0, len(features1)):
        if inlier_threshold(features1[i],features2[i],Fundamental_Matrix)<0.001:
            count = count + 1 
            temp1.append(features1[i])
            temp2.append(features2[i])
    if count > Inliers: 
        Inliers = count
        finalFund_Matrix = FundMatrix
        inlier1 = temp1
        inlier2 = temp2
    return inlier1, inlier2, finalFund_Matrix



def Essential_Matrix(K,F):
    E_temp= np.matmul(np.matmul(K.T,F),K)
    #singular values of E are not necessarily (1,1,0) due to the noise in K. 
    #This can be corrected by reconstructing it with (1,1,0) singular values,
    s = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    u, s, v = np.linalg.svd(E_temp, full_matrices=True)

    E_final=np.matmul(np.matmul(u,s),v.T)
    return E_final # 5 Dof (normalized image coordinates)


def Camera_Pose(essentialMatrix):
    u, s, v = np.linalg.svd(essentialMatrix, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R=[]
    C=[]

    C,append(u[:,3])
    R1=np.matmul(np.matmul(u,w),v.T)
    R.append(R1)

    C.append((-1)*u[:,3])
    R2=np.matmul(np.matmul(u,w),v.T)
    R.append(R2)

    C.append(u[:,3])
    R3=np.matmul(np.matmul(u,w.transpose()),v.T)
    R.append(R3)
  

    C.append((-1)*u[:,3])
    R4=np.matmul(np.matmul(u,w.tranpose()),v.T)
    R.append(R4)
    for i in range(len(R)):
        if np.linalg.det(R[i])== -1:
            R[i]=-R[i]
            C[i]=-C[i]
        else:
            R[i]=R[i]
            C[i]=C[i]
    return R, C

def rotationMatrixToEulerAngles(R) :
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])
# inputs- camera poses from frame 1 and frame 2. feature point from frame 1 and corresponding feature point from frame 2

def TriangulationPoint(C1, C2, f1, f2):
    x = np.array([[0, -1, f1[1]], [1, 0, -f1[0]], [-f1[1], f1[0], 0]])
    x_ = np.array([[0, -1, f2[1]], [1, 0, -f2[0]], [-f2[1], f2[0], 0]])
    A1 = np.matmul(x , C1[0:3, :] )
    A2 = np.matmul(x_ , C2)
    A_x = np.vstack((A1, A2))
    u, s, v = np.linalg.svd(A_x)
    new_X = v[-1]
    new_X = new_X/new_X[3]
    new_X = new_X.reshape((4,1))
    return new_X[0:3].reshape((3,1))

# it will take list of rotational matrices,camera poses and features from current and next frame and give disambigous pose.
#the best camera configuration, (C,R,X) is the one that produces the maximum number of points satisfying the cheirality condition.


def che_condition(R, C, features1, features2):
    check = 0
    H = np.identity(4) # current camera pose 
    for index in range(0, len(R)): # rotation matrices
        angles = rotationMatrixToEulerAngles(R[index]) # euler angles of the rotation matrix
        #print('angle', angles)
        
        # If the rotation of x and z axis are within the -50 to 50 degrees then it is considered 
        if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50: 
            count = 0 
            newP = np.hstack((R[index], C[index])) # New camera Pose 
            for i in range(0, len(features1)): # Looping over all the inliers
                temp1x = getTriangulationPoint(H[0:3,:], newP, features1[i], features2[i]) # Triangulating all the inliers
                thirdrow = R[index][2,:].reshape((1,3)) 
                if np.squeeze(np.matmul(thirdrow,(temp1x - Clist[index]))) > 0: # If the depth of the triangulated point is positive
                    count = count + 1 

            if count > check: 
                check = count
                C_ = C[index]
                R_= R[index]
                
    if C_[2] > 0:
        C_ = -C_
                
    return R_, C_
    
        
   
    

    
        










lastH = np.identity(4) # Initial camera pose
origin = np.array([[0, 0, 0, 1]]).T 



data_points = []
for index in range(19, 21):
    #print(frames[index], index)
    img_1 = cv2.imread("stereo/centre/" + str(frames[index]), 0)
    BGR_1 = cv2.cvtColor(img_1, cv2.COLOR_BayerGR2BGR)
    undist_1 = UndistortImage(BGR_1,LUT)  
    gray_1 = cv2.cvtColor(undist_1,cv2.COLOR_BGR2GRAY)

    img_2 = cv2.imread("stereo/centre/" + str(frames[index + 1]), 0)
    BGR_2 = cv2.cvtColor(img_2, cv2.COLOR_BayerGR2BGR)
    undist_2 = UndistortImage(BGR_2,LUT)  
    gray_2 = cv2.cvtColor(undist_2,cv2.COLOR_BGR2GRAY)
    # Reducing the  roi to eliminate extra things like sky, car bonnete,etc
    Image_1 = gray_1[200:650, 0:1280]
    Image_2 = gray_2[200:650, 0:1280]
    
    # orb = cv2.ORB_create(nfeatures=1500)
    # keypoints1_orb, descriptors_1 = orb.detectAndCompute(Image_1, None)
    # keypoints2_orb, descriptors_2 = orb.detectAndCompute(Image_2, None)
    # surf = cv2.xfeatures2d.SURF_create()
    # keypoints1_surf, descriptors_1= surf.detectAndCompute(Image_1, None)
    # keypoints2_surf, descriptors_2 = surf.detectAndCompute(Image_2, None)
    # img = cv2.drawKeypoints(Image_1, keypoints1_orb, None)
    # plt.imshow(img)
    
    # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # # Match descriptors.
    # matches = bf.match(descriptors_1,descriptors_2)
    # # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)
    # # Draw first 10 matches.
    # img3 = cv2.drawMatches(Image_1,keypoints1_orb,Image_2,keypoints2_orb,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()


    sift = cv2.xfeatures2d.SIFT_create()
#find the keypoints and descriptors with SIFT
    keypoints1_sift, descriptors_1 = sift.detectAndCompute(Image_1,None)
    keypoints2_sift, descriptors_2 = sift.detectAndCompute(Image_2,None)
    features1 = []
    features2 = []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)
    search_params = dict()   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_1,descriptors_2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]
    #ratio test 
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.1*n.distance:
            #matchesMask[i]=[1,0]
            features1.append(keypoints1_sift[m.queryIdx].pt)
            features2.append(keypoints2_sift[m.trainIdx].pt)
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(Image_1,keypoints1_sift,Image_2,keypoints2_sift,matches,None,**draw_params)
    #   plt.imshow(img3,),plt.show()
    f1, f2 =get_eights_points(features1,features2)

    Fundamental_Matrix = fundamentalMatrix(f1,f2)
    Inliers1,Inliers2, FinalFund_Matrix= get_inliers(features1,features2,Fundamental_Matrix)

    EssentialMatrix= Essential_Matrix(K,FinalFund_Matrix)

    #Rotational and translational matrices from essential matrix
    R, T = Camera_Pose(EssentialMatrix)

    # Computing all the solutions of rotation matrix and translation vector
   
    # Disambiguating one solution from four
    R_, T_ = che_condition(R, T, Inliers1, Inliers2) 

    lastH = np.matmul(lastH ,Homogenousmatrix(R_, T_) )# Transforming from current frame to next frame
    p = np.matmul(lastH,origin) # Determining the transformation of the origin from current frame to next frame

    #print('x- ', p[0])
    #print('y- ', p[2])
    data_points.append([p[0][0], -p[2][0]])
    plt.scatter(p[0][0], -p[2][0], color='r')
    
    if cv2.waitKey(0) == 27:
        break
        
cv2.destroyAllWindows()
#df = pd.DataFrame(l, columns = ['X', 'Y'])
#df.to_excel('test_code_last1.xlsx')
plt.show()


# In[ ]:


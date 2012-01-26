import numpy as np
from numpy.linalg import inv
from sympy import Symbol
from sympy.matrices import *

#see detail comments in hexahedra_4

x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])

#anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
#x2y2z2 are basis of new coor defined in the original frame,new=T.orig
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])

#f2 calculate the distance b/ p1 and p2
f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

#anonymous function f3 is to calculate the coordinates of basis with magnitude of 1.,p1 and p2 are coordinates for two known points, the 
#direction of the basis is pointing from p1 to p2
f3=lambda p1,p2:(1./f2(p1,p2))*(p2-p1)+p1

class share_face():
    def __init__(self,face=np.array([[0.,0.,0.],[0.5,0.5,0.5],[1.0,1.0,1.0]])):
        #pass in the vector of three known vertices
        self.face=face
    def share_face_init(self,flag='right_triangle'):
        #octahedra has a high symmetrical configuration,there are only two types of share face.
        #flag 'right_triangle' means the shared face is defined by a right triangle with two equal lateral and the other one
        #passing through body center;'regular_triangle' means the shared face is defined by a regular triangle
        p0,p1,p2=self.face[0,:],self.face[1,:],self.face[2,:]
        #consider the possible unregular shape for the known triangle
        dist_list=[np.sqrt(np.sum((p0-p1)**2)),np.sqrt(np.sum((p1-p2)**2)),np.sqrt(np.sum((p0-p2)**2))]
        index=dist_list.index(max(dist_list)) 
        
        if flag=='right_triangle':
        #'2_1'tag means 2 atoms at upside and downside, the other one at middle layer
            if index==0:self.center_point=(p0+p1)/2
            elif index==1:self.center_point=(p1+p2)/2
            elif index==2:self.center_point=(p0+p2)/2
            else:self.center_point=(p0+p2)/2
        elif flag=='regular_triangle':
            #the basic idea is building a sperical coordinate system centering at the middle point of each two of the three corner
            #and then calculate the center point through theta angle, which can be easily calculated under that geometrical seting
            def _cal_center(p1,p2,p0):
                origin=(p1+p2)/2
                y_v=f3(np.zeros(3),p1-origin)
                x_v=f3(np.zeros(3),p0-origin)
                z_v=np.cross(x_v,y_v)
                T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
                r=f2(p1,p2)/2.
                phi=0.
                theta=np.pi/2+np.arctan(np.sqrt(2))
                center_point_new=np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
                center_point_org=np.dot(inv(T),center_point_new)+origin
                #the two possible points are related to each other via invertion over the origin
                if abs(f2(center_point_org,p0)-f2(center_point_org,p1))>0.00001:
                    center_point_org=2*origin-center_point_org
                return center_point_org
            self.center_point=_cal_center(p0,p1,p2)
        self._find_the_other_three(self.center_point,p0,p1,p2,flag)
        
    def _find_the_other_three(self,center_point,p0,p1,p2,flag):
        dist_list=[np.sqrt(np.sum((p0-p1)**2)),np.sqrt(np.sum((p1-p2)**2)),np.sqrt(np.sum((p0-p2)**2))]
        index=dist_list.index(max(dist_list))
        
        if flag=='right_triangle':
            def _cal_points(center_point,p0,p1,p2):
                #here p0-->p1 is the long lateral
                z_v=f3(np.zeros(3),p2-center_point)
                x_v=f3(np.zeros(3),p0-center_point)
                y_v=np.cross(z_v,x_v)
                T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
                r=f2(center_point,p0)
                p3_new=np.array([r*np.cos(np.pi/2)*np.sin(np.pi/2),r*np.sin(np.pi/2)*np.sin(np.pi/2),0])
                p4_new=np.array([r*np.cos(3*np.pi/2)*np.sin(np.pi/2),r*np.sin(3*np.pi/2)*np.sin(np.pi/2),0])
                p3_old=np.dot(inv(T),p3_new)+center_point
                p4_old=np.dot(inv(T),p4_new)+center_point
                p5_old=2*center_point-p2
                return T,r,p3_old,p4_old,p5_old
            if index==0:#p0-->p1 long lateral
                self.T,self.r,self.p3,self.p4,self.p5=_cal_points(center_point,p0,p1,p2)
            elif index==1:#p1-->p2 long lateral
                self.T,self.r,self.p3,self.p4,self.p5=_cal_points(center_point,p1,p2,p0)
            elif index==2:#p0-->p2 long lateral
                self.T,self.r,self.p3,self.p4,self.p5=_cal_points(center_point,p0,p2,p1)
        elif flag=='regular_triangle':
            x_v=f3(np.zeros(3),p2-center_point)
            y_v=f3(np.zeros(3),p0-center_point)
            z_v=np.cross(x_v,x_v)
            self.T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
            self.r=f2(center_point,p0)
            self.p3=2*center_point-p0
            self.p4=2*center_point-p1
            self.p5=2*center_point-p2
             
    def cal_point_in_fit(self,r,theta,phi):
        #during fitting,use the same coordinate system, but a different origin
        #note the origin_coor is the new position for the sorbate0, ie new center point
        x=r*np.cos(phi)*np.sin(theta)
        y=r*np.sin(phi)*np.sin(theta)
        z=r*np.cos(theta)
        point_in_original_coor=np.dot(inv(self.T),np.array([x,y,z]))+self.center_point
        return point_in_original_coor

class share_edge(share_face):
    def __init__(self,edge=np.array([[0.,0.,0.],[5,5,5]])):
        self.edge=edge
        
    def cal_p2(self,ref_p=None,theta=0.,phi=np.pi/2,flag='off_center',**args):
        p0=self.edge[0,:]
        p1=self.edge[1,:]
        origin=(p0+p1)/2
        dist=f2(p0,p1)
        diff=p1-p0
        c=np.sum(p1**2-p0**2)
        ref_point=0
        if ref_p!=None:
            ref_point=ref_p
        elif:
            x,y,z=0.,0.,0.
            #set the reference point as simply as possible,using the same distance assumption, we end up with a plane equation
            #then we try to find one cross point between one of the three basis and the plane we just got
            #here combine two line equations (ref-->p0,and ref-->p1,the distance should be the same)
            if diff[0]!=0:
                x=c/(2*diff[0])
            elif diff[1]!=0.:
                y=c/(2*diff[1])
            elif diff[2]!=0.:
                z=c/(2*diff[2])
            ref_point=np.array([x,y,z])
            if sum(ref_point)==0:
                #if the vector (p0-->p1) pass through origin [0,0,0],we need to specify another point satisfying the same-distance condition
                #here, we a known point (x0,y0,z0)([0,0,0] in this case) and the normal vector to calculate the plane equation, 
                #which is a(x-x0)+b(y-y0)+c(z-z0)=0, we specify x y to 1 and 0, calculate z value.
                #a b c coresponds to vector origin-->p0
                ref_point=[1.,0.,-p0[0]/p0[2]]
        if flag=='cross_center':
            x1_v=f3(np.zeros(3),ref_point-origin)
            z1_v=f3(np.zeros(3),p1-origin)
            y1_v=np.cross(z1_v,x1_v)
            T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
            r=dist/2
            x_p2=r*np.cos(phi)*np.sin(np.pi/2)
            y_p2=r*np.sin(phi)*np.sin(np.pi/2)
            z_p2=0
            p2_new=np.array([x_p2,y_p2,z_p2])
            p2_old=np.dot(inv(T),p2_new)+origin
            self.p2=p2_old
            self.face=np.append(self.edge,[p2_old],axis=0)
            self.flag='right_triangle'
        elif flag=='off_center':
            z1_v=f3(np.zeros(3),ref_point-origin)
            x1_v=f3(np.zeros(3),p1-origin)
            y1_v=np.cross(z1_v,x1_v)
            T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
            r=dist/2.
            #note in this case, phi can be either pi/2 or 3pi/2, theta can be any value in the range of [0,pi]
            x_center=r*np.cos(phi)*np.sin(theta)
            y_center=r*np.sin(phi)*np.sin(theta)
            z_center=r*np.cos(theta)
            center_org=np.dot(inv(T),np.array([x_center,y_center,z_center]))+origin
            p2_old=2*center_org-p0
            self.p2=p2_old
            self.face=np.append(self.edge,[p2_old],axis=0)
            self.flag='right_triangle'
            
class share_corner(share_edge):
    #if want to share none, then just set the corner coordinate to the first point set arbitratly.
    def __init__(self,corner=np.array([0.,0.,0.])):
        self.corner=corner
        
    def cal_p1(self,r,theta,phi):
        #here we simply use the original coordinate system converted to spherical coordinate system, but at different origin
        x_p1=r*np.cos(phi)*np.sin(theta)+self.corner[0]
        y_p1=r*np.sin(phi)*np.sin(theta)+self.corner[1]
        z_p1=r*np.cos(theta)+self.corner[2]
        p1=np.array([x_p1,y_p1,z_p1])
        self.p1=p1
        self.edge=np.append(self.corner,[p1],axis=0)

if __name__=='__main__':
    test1=octahedra_2.share_edge(edge=np.array([[0.,0.,0.],[5.,5.,5.]]))
    test1.cal_p2(theta=0,phi=np.pi/2,flag='cross_center')
    test1.share_face_init(flag=test1.flag)
    print test1.face,test1.p3,test1.p4,test1.p5,test1.center_point

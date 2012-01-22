import numpy as np
from numpy.linalg import inv
from sympy import Symbol
from sympy.matrices import *
#see detail comment in hexahedra_4
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
        self.face=face
        
    def share_face_init(self,**args):
        p0,p1,p2=self.face[0,:],self.face[1,:],self.face[2,:]
        #consider the possible unregular shape for the known triangle
        dist_list=[np.sqrt(np.sum((p0-p1)**2)),np.sqrt(np.sum((p1-p2)**2)),np.sqrt(np.sum((p0-p2)**2))]
        index=dist_list.index(max(dist_list)) 
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])
        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)
        z_v=f3(np.zeros(3),np.cross(p1-center_point,p0-center_point))
        x_v=f3(np.zeros(3),p1-center_point)
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        self.T=T
        r=f2(p0,center_point)
        r_bc=r*(np.sqrt(2.)/4.)
        r_ed=r*(np.sqrt(2.))
        body_center_new=np.array([0.,0.,r_bc*np.cos(0.)])
        body_center_old=np.dot(inv(T),body_center_new)+center_point
        p3_new=np.array([0.,0.,r_ed*np.cos(0.)])
        p3_old=np.dot(inv(T),p3_new)+center_point
        self.p3,self.center_point,self.r=p3_old,body_center_old,f2(body_center_old,p0)
        
    def cal_point_in_fit(self,r,theta,phi):
        #during fitting,use the same coordinate system, but a different origin
        #note the origin_coor is the new position for the sorbate0, ie new center point
        x=r*np.cos(phi)*np.sin(theta)
        y=r*np.sin(phi)*np.sin(theta)
        z=r*np.cos(theta)
        point_in_original_coor=np.dot(inv(self.T),np.array([x,y,z]))+self.center_point
        return point_in_original_coor

class share_edge(share_face):
    def __init__(self,edge=np.array([[0.,0.,0.],[0.5,0.5,0.5]])):
        self.edge=edge
        self.flag=None
        
    def cal_p2(self,theta=0.,phi=np.pi/2,**args):
        p0=self.edge[0,:]
        p1=self.edge[1,:]
        origin=(p0+p1)/2
        dist=f2(p0,p1)
        diff=p1-p0
        c=np.sum(p1**2-p0**2)
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
            ref_point=[1.,0.,-p0[0]/p0[2]]
        x1_v=f3(np.zeros(3),ref_point-origin)
        y1_v=f3(np.zeros(3),p1-origin)
        z1_v=np.cross(x1_v,y1_v)
        T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
        #note the r is different from that in the case above
        #note in this case, phi can be either pi/2 or 4pi/3, theta can be any value in the range of [0,pi]
        r=dist/2*np.sqrt(3.)
        x_p2=r*np.cos(phi)*np.sin(theta)
        y_p2=r*np.sin(phi)*np.sin(theta)
        z_p2=r*np.cos(theta)
        p2_new=np.array([x_p2,y_p2,z_p2])
        p2_old=np.dot(inv(T),p2_new)+origin
        self.p2=p2_old
        self.face=np.append(self.edge,[p2_old],axis=0)
        
class share_corner(share_edge):
#if want to share none, then just set the corner coordinate to the first point arbitratly.
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
    test1=tetrahedra_3.share_edge(edge=np.array([[0.,0.,0.],[5.,5.,5.]]))
    test1.cal_p2(theta=0,phi=np.pi/2)
    test1.share_face_init()
    print test1.face,test1.p3,test1.center_point
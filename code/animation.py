import os
import utils
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from models import EncounterabilityKNN    


class AnimatedScatter(object):
    def __init__(self,I,scale=30,num=0,save=True):
        self.scale = scale
        self.set_up_init(I)
        self.anim = FuncAnimation(self.fig, self.update, fargs=(I,), frames=np.arange(11), interval=200)
        self.save_fig(num, save)
        
    def set_up_init(self,I):
        self.fig, self.ax = plt.subplots()
        self.num_cars = int(utils.len_notnull(I)/2)    
        
        self.le = self.num_cars*2
        
        self.age_scat = self.ax.scatter([I[0,0]],[I[0,1]],c='r')
        self.oth_scat = self.ax.scatter([I[0,2:self.le:2]],[I[0,3:self.le:2]],c='b')


        
        self.trajs =[None]*self.num_cars
        for i in range(self.num_cars):
            self.trajs[i] = self.ax.plot(I[0,2*i],I[0,2*i+1],lw=2)

        #self.ax.axis([np.min(I[:,0::2]), np.max(I[:,0::2]), np.min(I[:,1::2]), np.max(I[:,1::2])])
        scale=self.scale
        self.ax.axis([-scale,scale,-scale,scale])
        self.ax.set_autoscale_on(False)
        
    def update(self, t,I):
        
        temp = np.zeros((self.num_cars-1,2))
        temp[:,0] =I[t,2:self.le:2]
        temp[:,1] =I[t,3:self.le:2]
        self.age_scat.set_offsets((I[t,0],I[t,1]))
        self.oth_scat.set_offsets(temp)
        
        self.trajs[0] = self.ax.plot(I[0:t,0],I[0:t,1],'r-',lw=2)
        for i in range(1,self.num_cars):   
            self.trajs[i] = self.ax.plot(I[0:t,2*i],I[0:t,2*i+1],'b-',lw=2)
        
        dr,dtan = utils.dynamics_data(I,t)
        
        
        label = 'timestep {0}, '.format(-1000+t*100)+'age_dr= %.3f, '%dr + 'age_dtan= %.3f, '%dtan
        self.ax.set_xlabel(label)
        return self.ax
    
    def save_fig(self,num,save):
        if save == True:
            plotname='Intersection_'+str(num)+'.mp4'
            fname = os.path.join("..",'figs',plotname)
            self.anim.save(fname, dpi=144, fps=2, extra_args=['-vcodec', 'libx264'])
        else:
            plt.show()
        plt.cla()
        plt.clf()
        plt.close('all')

        
class AnimatedAnalysis(AnimatedScatter):
    def __init__(self,I,ind_knn,scale=30,num=0,save=True,anim_label=False):
        self.I = I
        self.scale = scale
        self.ind_knn=ind_knn
        self.anim_label=anim_label
        
        self.set_up_init()
        self.anim = FuncAnimation(self.fig, self.update, fargs=(I,), frames=np.arange(11), interval=200)
        self.save_fig(num, save)
    
        
    def set_up_init(self):
        self.fig, ((self.ax1,self.ax2),(self.ax3,self.ax4)) = plt.subplots(2,2)
        self.num_cars = int(utils.len_notnull(self.I)/2)    
        self.le = self.num_cars*2
        
        self.draw_points_init()
        self.draw_data_init()
        self.draw_traj_init()
        self.filtered_proj_init()    
    
    def draw_points_init(self):
        #I_knn=utils.I_inds(self.I,self.ind_knn)
        age_points_size=20
        oth_points_size=20
        T_points_size=5
        
        self.ax1.set_title("animation_intersection")
        self.age_scat = self.ax1.scatter([self.I[0,0]],[self.I[0,1]],c='r')
        age_sizes=np.ones(1)
        self.age_scat.set_sizes(age_sizes*age_points_size)
        
        if self.num_cars >1:
            self.oth_scat = self.ax1.scatter([self.I[0,2:self.le:2]],[self.I[0,3:self.le:2]],c='b')
            oth_sizes=np.ones(self.num_cars-1)
            self.oth_scat.set_sizes(oth_sizes*oth_points_size)
        
        
        self.xy_T=EncounterabilityKNN(self.I,self.I).Coor_T_knn_tavg(self.I,self.ind_knn)[0]
        T_sizes=np.ones(self.xy_T.shape[0])
        self.T = self.ax1.scatter(self.xy_T[:,0],self.xy_T[:,1],c='g')
        self.T.set_sizes(T_sizes*T_points_size)
        
        scale=self.scale
        self.ax1.axis([-scale,scale,-scale,scale])
        self.ax1.set_autoscale_on(False)

    def draw_data_init(self):
        r_t_age,dr_t_age,ddr_t_age=utils.dynamics_time_data(self.I,0)[3:]
        self.plot_xt(self.ax2,dr_t_age)
        self.ax2.set_title("dr_t_age")
        self.plot_xt(self.ax3,ddr_t_age)
        self.ax3.set_xlabel("ddr_t_age")
        
        #self.plot_xt(self.ax4,tavg_ddr)
        
        #Plot the final data analyasis plot in terms of statstical information
        num_knn=len(self.ind_knn)
        tavg_ddr = ddr_t_age.mean()
        
        #Range functions
        range_num=np.arange(0,10,1)
        ymin=-0.02
        ymax=0.02
        range_y=np.arange(ymin,ymax,(ymax-ymin)/10)
        ones=np.ones(len(range_num))
        
        self.ax4.plot(range_num,ones*0,'k-',lw=0.5)
        self.ax4.plot(range_num,ones*tavg_ddr,'r-',lw=1)        
        self.ax4.plot(ones*num_knn,range_y,'g-',lw=1)
        #self.ax4.plot([num_knn],[tavg_ddr],'bo')

        
        self.ax4.set_ylim(ymin,ymax)
        self.ax4.set_xlabel("tavg_ddr=%.4f, knn= %d"%(tavg_ddr,len(self.ind_knn)))
    
    def draw_traj_init(self):
        self.trajs =[None]*self.num_cars
        for i in range(self.num_cars):
            self.trajs[i] = self.ax1.plot(self.I[0,2*i],self.I[0,2*i+1],lw=2)
    
    def filtered_proj_init(self):
        # ind_oth = ind_knn_filter(self.I,self.ind_knn)
        # xy_T_new = Coor_T_knn_tavg(self.I,ind_oth)[0]
        # I_new = utils.I_inds(self.I,ind_oth)
        # I_oth =I_new[:,2:]
        I_oth = utils.I_inds(self.I,self.ind_knn)
        for i in range(len(self.ind_knn)):
            x_T= self.xy_T[i,0]
            y_T= self.xy_T[i,1]
            x_oth = I_oth[0,2*i]
            y_oth = I_oth[0,2*i+1]
            # print(np.array([x_T,x_oth]))
            # print(np.array([y_T,y_oth]))
            self.projs =self.ax1.plot(np.array([x_T,x_oth]),np.array([y_T,y_oth]),'g--')
        scale=self.scale
        self.ax1.plot(np.arange(-scale,scale,1),np.arange(-scale,scale,1)*0,'k-',lw=0.5)
        
    def update(self, t,I):
        
        #Update the location of agent
        self.age_scat.set_offsets((I[t,0],I[t,1]))
        #Update the trajectories of cars.
        self.trajs[0] = self.ax1.plot(I[0:t,0],I[0:t,1],'r-',lw=2)
        if self.num_cars >1:
            #Update the location of other cars
            temp = np.zeros((self.num_cars-1,2))
            temp[:,0] =I[t,2:self.le:2]
            temp[:,1] =I[t,3:self.le:2]
            self.oth_scat.set_offsets(temp)
        for i in range(1,self.num_cars):   
            self.trajs[i] = self.ax1.plot(I[0:t,2*i],I[0:t,2*i+1],'b-',lw=2) 
        
        dr,dtan = utils.dynamics_data(I,t)
        
        if self.anim_label:
            label = 'timestep {0}, '.format(-1000+t*100)+'age_dr= %.3f, '%dr + 'age_dtan= %.3f, '%dtan
            self.ax1.set_xlabel(label)
            
        return self.ax1
    
    def plot_xt(self,ax,x):
        ax.plot(utils.range_t(x),x,'ko-')


import pygame, sys
import numpy as np
from scipy.integrate import ode

# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
SKYBLUE = (0, 204, 255)
PURPLE = (255,0,255)
g = -9.8

WIDTH = 800
HEIGHT = 640


air_coef =  0.1 # coef of frition for air


class SlingShot(pygame.sprite.Sprite):
    def __init__(self,name,pos,k,l,maxl):
        super().__init__()
        self.name= name # name used to Identify slingshot
        self.image = pygame.image.load("finalProject/images/slingShot.png").convert_alpha()
        self.image = pygame.transform.scale_by(self.image,0.06)
        self.rect = self.image.get_rect()
        self.pos = pos # position graphic
        self.hingeLoc = [pos[0]+50,pos[1]-30]# position of hinge 
        self.l = l # rest length of slingshot
        self.maxL = maxl # max length of sling before it deactivates
        self.k = k # spring constant for slingshot
        self.activationRange = (maxl + self.l)/2 # length at which slingshot will activate.

class RigidBall(pygame.sprite.Sprite):
    def __init__(self,name, radius,mass,pos1,vel1,color):
        super().__init__()
        self.name = name # identifier
        self.r = radius # radius of ball
        self.mass = mass # mass of ball
        self.mk = 100 # mouse spring constant

        self.CSlist = [] # list to store connect slingshots


        self.collision_dealt = False # make sure there arent multiple collisions
        self.image = pygame.Surface([radius*2, radius*2],pygame.SRCALPHA).convert_alpha()
        self.image.set_colorkey((0, 0, 0))

        pygame.draw.circle(self.image, color, (radius, radius), radius, radius)

        self.image = self.image.convert_alpha()
        self.image_rot = self.image 

        self.isClicked = False # bool that checks if the ball is clicked

        self.rect = self.image.get_rect()
        self.rect.center = self.rect.center

        self.t = 0 # sim time
        self.pos = np.array([0.,0.]) # position of object (only used to pass to pygame)

        self.state = np.zeros(4) 
        self.state[0:2] = pos1 # position of ball (used for calculations)
        self.state[2:4] = vel1 # velocity of ball

        
        self.solver = ode(self.f) # ODE solver
        self.solver.set_integrator('dop853')
        self.solver.set_f_params(0.,0.)
        self.solver.set_initial_value(self.state, self.t)

    # detects collisions between ball and floor
    def collision_detect(self,state):
        return state[1] <= -270+self.r # if the ball under the floor + its radius it collides with the floor.
    
    # respond to collision between ball and floor
    def collision_respond(self,state,t,F_net):
        t1 = self.t # current sim time
        t2 = t # time after collision

        i = 0 # makes sure loop later on doesnt run forever
        acc = F_net/self.mass # acceration at t1

        cur_pos = self.state[0:2] # position at current sim time
        cur_vel = self.state[2:4] # veloctiy at current sim time

        new_vel = state[2:4] # velocity after collision
        new_pos = state[0:2] # position of ball as after collision
        new_dt = 0 # variable for change in time between current time and actual time of collision
        error = abs(new_pos[1]-(-270)+self.r) # how far the ball is from the ground.

        while(error > 0.00000001):# loop runs until we are acceptably close to the floor 
            mid_t = (t1+t2)/2.0 # time halfway between the current time and collision time
            new_dt = mid_t - self.t # time change between the current time and halfway point.

            new_pos = cur_pos + cur_vel * new_dt # position at halfway point
            new_vel = cur_vel + acc * new_dt # velocity at halfway point
            # if the ball is still under floor at the halfway point we move the time after the collison to the halfway point.
            if new_pos[1] < (-270+self.r):
                t2 = mid_t
            else:# else we move the time before the collison to the halfway point
                t1 = mid_t
            
            error = abs(new_pos[1]-(-270+self.r)) #recaclulate error

            if i == 1000: # if look runs more than 50 time break
                break   # note if one of the balls comes to rest this can freeze the simulaions
            i+=1
        # update
        state[0:2] = new_pos 
        state[2:4] = new_vel
        state[3] = -1 *state[3]
        
        return state, self.t + new_dt # return state and time
    
    def ballColdetect(self,ball):
        d = self.pos - ball.pos  # vector between center of both balls
        dist = np.linalg.norm(d) # distance between balls

        if dist < self.r+ball.r: # check that dist between two balls is lesser then the sum of their radii if true the ball colliding
            return True
        else:
            return False   


    def ballColResp(self,ball):
        c1 = self.state[0:2] # center of current ball
        c2 = ball.state[0:2] # center of ball we colliding with

        m1 = self.mass # mass of current ball
        m2 = ball.mass # mass of ball we are colliding with

        v1 = self.state[2:4] # velocity of current ball
        v2 = ball.state[2:4] # velocity of ball we are colliding with
        M = self.mass + ball.mass # sum  mass of both balls

        # new velocities
        v1hat = v1 - (((2*m2)/(M))*((np.dot(v1-v2,c1-c2))/(np.linalg.norm(c1-c2)**2))*(c1-c2))
        v2hat = v2 - ((((2*m1)/M))*((np.dot(v2-v1,c2-c1))/(np.linalg.norm(c2-c1)**2))*(c2-c1))

        print(ball.name,np.linalg.norm(ball.state[2:4])," : ",np.linalg.norm(v2hat))
        print(self.name,np.linalg.norm(self.state[2:4])," : ",np.linalg.norm(v1hat))

        if self.collision_dealt == False: # check we dont collide with ball twice in the same instance
            self.state[2:4] = v1hat
            ball.state[2:4] = v2hat

        self.collision_dealt = ball.collision_dealt = True # set collison dealt to True to we dont collide with the ball agian on a different pass.
    # ODE solver Function
    def f(self, t, state, force):
        rate = np.zeros(4)
        rate[0:2] = state[2:4] 
        rate[2:4] = force/self.mass

        return rate
    
    # tells if a slingshot is connected to a ball
    def SlingConnected(self,sling):
        d = self.state[0:2] - sling.hingeLoc
        dist = np.linalg.norm(d) # dist between ball and hinge of slingshot
        t = sling.name in self.CSlist # if ball has this slingshot is in the list of connected slingshot
        if dist < sling.activationRange and self.isClicked: # if ball is in activation zone of slingshot and clicked the spring is added to list of connected spring and connect bool is set to true
            t = True
            self.CSlist.append(sling.name)

        if dist < sling.l or dist > sling.activationRange and t==True:# if ball is outside activation range or in dead zone of sling shot it is realeased and the slingshot list is cleared
            t = False
            self.CSlist.clear()
        return t


    def update(self,dt,screen,balls,slings):
        F_s = np.array([0,0]) # force of slingshot/slinghots on ball
        F_m = np.array([0,0]) # force of mouse spring on ball.
        F_g = np.array([0,g*self.mass]) # force of gravity 
        F_Ar = -air_coef * self.state[2:4] # force of air resistance
        F_net = 0 # total force

        #check if ball collides with any other balls
        for ball in balls:
            if ball.name!=self.name:
                if self.ballColdetect(ball):
                    self.ballColResp(ball)

        # checks if we are connected to any slingshot
        for sling in slings:
            if self.SlingConnected(sling):
                d = self.state[0:2] - sling.hingeLoc
                dist = np.linalg.norm(d)
                displacment = dist - sling.l
                u = d/dist
                F_s = F_s + (-sling.k *  displacment * u)
                self.draw_line(screen,sling.hingeLoc)
                
        # adds mouse spring force if ball is in clicked state
        if self.isClicked:
            hinge = self.to_system(pygame.mouse.get_pos())
            d1 = self.state[0:2] - hinge
            r = np.linalg.norm(d1)
            u1 = d1/r
            displacment1 = r 
            F_m = (-self.mk*self.mass *  displacment1)*u1
            self.state[2:4] = np.array([0,0])

        
        F_net = F_g + F_Ar + F_s + F_m #calculate total force

        # pass new force total force to ODE solver
        self.solver.set_f_params(F_net)
        new_state = self.solver.integrate(self.t + dt)# get update
        # check if ball has collided with floor
        if not self.collision_detect(new_state) == True:
            self.state = new_state
            self.t += dt
        else:
            col_state,col_time = self.collision_respond(new_state,self.t+dt,F_net)
            self.state = col_state
            self.t = col_time
        # edge case: if the ball goes off screen we move it to the other edge.
        if self.state[0]>WIDTH/2:
            self.state[0] = -WIDTH/2
        elif self.state[0]<-WIDTH/2:
            self.state[0] = WIDTH/2

        self.collision_dealt = False
        self.pos = self.state[0:2]

    def draw_line(self,screen,pos1): #draws line between center of ball and some point

        pygame.draw.line(screen,BLACK,self.to_screen([self.pos[0],self.pos[1]]),self.to_screen(pos1),3)
        pygame.display.update()

    def to_system(self,mpos):#converts pygame coords to system coords
        return [int(((mpos[0])-WIDTH//2)), int(((-mpos[1])+HEIGHT//2))]
    
   
    def to_screen(self, pos):#converts system coords to pygame coords
        return [int((pos[0])+WIDTH//2), int((-1*pos[1])+HEIGHT//2)] 



# just store ground graphic
class Ground(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("finalProject/images/ground.png").convert_alpha()
        self.image = pygame.transform.scale_by(self.image,0.5)
        self.rect = self.image.get_rect()
        self.pos = [-420.,-270.,0.]


class World: # systems class
    def __init__(self):
        self.balls = pygame.sprite.Group() # holds all ball objects in system
        self.ground = pygame.sprite.Group() # holds all slingshot objects in system
        self.slings = pygame.sprite.Group() # holds all ground objects in system
        self.i=0

    def addBall(self,body):
        self.balls.add(body)
    
    def addGround(self,body):
        self.ground.add(body)
    

    def addSling(self, body):
        self.slings.add(body)

    def update(self,dt,screen):
        
        # output ground
        for obj in self.ground:
            p = self.to_screen(obj.pos)
            obj.rect.x, obj.rect.y = p[0], p[1] 
        # output slingshots
        for obj in self.slings:
            p = self.to_screen(obj.pos)
            obj.rect.x, obj.rect.y = p[0], p[1] 
        
        # update and output balls
        for obj in self.balls:

            p = self.to_screen(obj.pos)
            # checks if any balls are clicked or realeased
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for ball in self.balls:
                        if ball.rect.collidepoint(event.pos):
                            ball.isClicked = True
                
                if event.type == pygame.MOUSEBUTTONUP:
                    for ball in self.balls:
                        ball.isClicked = False


            obj.update(dt,screen,self.balls,self.slings)  
            obj.rect.x, obj.rect.y = p[0]-obj.r, p[1]-obj.r
            # checks if any balls are clicked or realeased
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for ball in self.balls:
                        if ball.rect.collidepoint(event.pos):
                            ball.isClicked = True
                
                if event.type == pygame.MOUSEBUTTONUP:
                    for ball in self.balls:
                        ball.isClicked = False
      

    
    def to_screen(self, pos):
        return [int((pos[0])+WIDTH//2), int((-1*pos[1])+HEIGHT//2)]
    

    def draw(self, screen):
        self.ground.draw(screen)
        self.slings.draw(screen)
        self.balls.draw(screen)

def main():
    pygame.init()
    win_width = WIDTH
    win_height = HEIGHT
    screen = pygame.display.set_mode((win_width, win_height))  # Top left corner is (0,0)
    pygame.display.set_caption('Final Project')

    run = True



    ball = RigidBall("ball1",30,5.,[-100,-200],[0,0],RED)
    ball2 = RigidBall("ball2",30,10.,[0,-200],[0,0],WHITE)
    ball3 = RigidBall("ball3",20,10.,[200,-200],[0,0],GREEN)
    ball4 = RigidBall("ball4",25,10.,[100,-200],[0,0],PURPLE)
    sling = SlingShot("s1",[-250.,-100.],10,20,250)
    sling2 = SlingShot("s2",[-200.,-100.],10,20,250)
    ground = Ground()


    theWorld = World()


    theWorld.addGround(ground)
    theWorld.addSling(sling)
    # theWorld.addSling(sling2)
    theWorld.addBall(ball2)
    theWorld.addBall(ball)
    # theWorld.addBall(ball3)
    # theWorld.addBall(ball4)

    dt = 0.02

    while run:
        screen.fill(SKYBLUE)

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit(0)

        theWorld.draw(screen)
        theWorld.update(dt,screen)
        pygame.display.flip()



if __name__ == '__main__':
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:52:32 2022

@author: namankhurpia
"""

# Imports
import cv2
import gym
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np



# Defining the Marios Brother's World Environment.
class MariosBros(gym.Env):
    
    def __init__(self, env_type):
       
        self.env_type = env_type

        self.env_width = 10
        
        self.env_height = 10

        self.number_of_agents = 2
        
        self.treasure_quantity = 1
        
        self.jump_pos = np.asarray([
                                    [7, 3], 
                                    [2, 4], [6, 4], [8, 4],
                                    [1, 5], 
                                    [3, 5],[7, 5],
                                    [2, 6],
                                    [4, 8], [8, 8],
                                    [3, 9], [5, 9], [7, 9], [9, 9],
                                    ])
        
        
        self.one_up_pos = np.asarray([
                                    [3, 0], [5, 0], 
                                    [8, 2], 
                                    [0, 3],
                                    [5, 4],
                                    [9, 6],
                                    [2, 7],
                                    [0, 9]
                                    ])
        
        self.plant_pos = np.asarray([[2, 0], [4, 0], [6, 0], 
                                    [3, 1], [5, 1], [8, 1], 
                                    [0, 2], [7, 2], [9, 2],
                                    [1, 3], [5, 3], [8, 3], 
                                    [0, 4], [4, 4], [6, 4],
                                    [5, 5], [9, 5], 
                                    [2, 6], [8, 6],  
                                    [1, 7], [3, 7], [9, 7],
                                    [0, 8], [2, 8],
                                    [1, 9]
                                    ])
        
        
        self.tortoise_pos = np.asarray([
                                    [7, 4], 
                                    [2, 5], 
                                    [4, 9], 
                                    [8, 9]
                                    ])
        
        #defining agent start position
        #self.car_pos = np.asarray([0, 0])
        self.mario_pos = np.asarray([0, 0])
        self.luigi_pos = np.asarray([9, 0])
        
        
        #self.treasure_pos = np.asarray([9, 8])
        self.princess_pos = np.asarray([9, 8])
        
        #keeping track of steps taken by agent
        self.timesteps = 0
        
        #keeping a max count of 100 steps taken by agent
        self.max_timesteps = 100

        self.coordinates_state_mapping = {}
        for i in range(self.env_height):
            for j in range(self.env_width):
                self.coordinates_state_mapping[f'{np.asarray([j, i])}'] = i * self.env_width + j

    def step(self, action_mario ,action_luigi):
        #mario going to right
        if action_mario == 0:
            self.mario_pos[0] = self.mario_pos[0] + 1  
        if action_luigi == 0:
            self.luigi_pos[0] = self.luigi_pos[0] + 1
            
        #mario going to left
        if action_mario == 1:
            self.mario_pos[0] = self.mario_pos[0] - 1  
        if action_luigi == 1:
            self.luigi_pos[0] = self.luigi_pos[0] - 1
            
        #mario going up  
        if action_mario == 2:
            self.mario_pos[1] = self.mario_pos[1] + 1  
        if action_luigi == 2:
            self.luigi_pos[1] = self.luigi_pos[1] + 1
            
        #mario going down    
        if action_mario == 3:
            self.mario_pos[1] = self.mario_pos[1] - 1  
        if action_luigi == 3:
            self.luigi_pos[1] = self.luigi_pos[1] - 1


        #clipping the agent's new position in case it tries to move out of the grid world
        self.mario_pos = np.clip(self.mario_pos, a_min=[0, 0],a_max=[self.env_width - 1, self.env_height - 1])
        self.luigi_pos = np.clip(self.luigi_pos, a_min=[0, 0],a_max=[self.env_width - 1, self.env_height - 1])
        
        observation_mario = self.coordinates_state_mapping[f'{self.mario_pos}']
        observation_luigi = self.coordinates_state_mapping[f'{self.luigi_pos}']

        self.timesteps = self.timesteps + 1

        reward_mario = 0
        reward_luigi = 0
       
        if np.array_equal(self.mario_pos, self.princess_pos) and self.treasure_quantity > 0:
            self.treasure_quantity = self.treasure_quantity - 1
            reward_mario = 100
        
        if np.array_equal(self.luigi_pos, self.princess_pos) and self.treasure_quantity > 0:
            self.treasure_quantity = self.treasure_quantity - 1
            reward_luigi = 100

        for i in range(len(self.jump_pos)): 
            if np.array_equal(self.mario_pos, self.jump_pos[i]):
                reward_mario = -10
            if np.array_equal(self.luigi_pos, self.jump_pos[i]):
                reward_luigi = -10


        if np.array_equal(self.mario_pos, self.tortoise_pos):
            reward_mario = -100
         
        if np.array_equal(self.luigi_pos, self.tortoise_pos):
            reward_luigi = -100

        if self.treasure_quantity == 0 or np.array_equal(self.mario_pos, self.tortoise_pos):
            done_mario = True
        else:
            done_mario = False
            
        if self.treasure_quantity == 0 or np.array_equal(self.luigi_pos, self.tortoise_pos):
            done_luigi = True
        else:
            done_luigi = False    
            
            
        for i in range(len(self.one_up_pos)):
            if np.array_equal(self.mario_pos, self.one_up_pos[i]):
                done_mario = True
            if np.array_equal(self.luigi_pos, self.one_up_pos[i]):
                done_luigi = True
        
        if self.timesteps == self.max_timesteps:
            done_mario = True
            done_luigi = True

        info = {}

        return observation_mario, observation_luigi, reward_mario, reward_luigi, done_mario, done_luigi, info


    def reset(self):
        #assigning default values to the environment and agent and all global variables
        #method returns an observational space
        self.mario_pos = np.asarray([0, 0])  
        self.luigi_pos = np.asarray([9, 0])  


        observation_mario = self.coordinates_state_mapping[f'{self.mario_pos}']
        observation_luigi = self.coordinates_state_mapping[f'{self.luigi_pos}']
        
        self.timesteps = 0  
        
        self.treasure_quantity = 1  

        return observation_mario,observation_luigi


    def render(self, mode='car', plot=False):
        
        fig, ax = plt.subplots(figsize=(25, 25)) 
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        def plot_image(plot_pos):
           
            #plot_car, plot_sign, plot_treasure, plot_pump, plot_uphill, plot_volcano = False, False, False, False, False, False
            plot_mario, plot_luigi, plot_jump, plot_princess, plot_one_up, plot_tortoise, plot_plant = False, False, False, False, False, False, False

           
            if np.array_equal(self.mario_pos, plot_pos):
                plot_mario = True
            if np.array_equal(self.luigi_pos, plot_pos):
                plot_luigi = True
                
                
            if any(np.array_equal(self.plant_pos[i], plot_pos) for i in range(len(self.plant_pos))):
                plot_plant = True
                
                
            if self.treasure_quantity > 0:  
                if np.array_equal(plot_pos, self.princess_pos):
                    plot_princess = True
                    
            if any(np.array_equal(self.one_up_pos[i], plot_pos) for i in range(len(self.one_up_pos))):
                plot_one_up = True
                
            if any(np.array_equal(self.jump_pos[i], plot_pos) for i in range(len(self.jump_pos))):
                plot_jump = True
                
            if any(np.array_equal( self.tortoise_pos[i], plot_pos) for i in range(len(self.tortoise_pos))):
                plot_tortoise = True
                
            
            #fulllist = [plot_mario,plot_luigi, plot_plant, plot_princess, plot_one_up, plot_jump, plot_tortoise])
            
            #print('plot mario',plot_mario , ',plot luigi:',plot_luigi)
            
            # Plot for mario.
            if plot_mario and \
                    all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_one_up, plot_jump, plot_tortoise]):
                mario = AnnotationBbox(OffsetImage(plt.imread('./images/mario.png'), zoom=0.28),np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(mario)
                
                if plot_luigi and \
                        all(not item for item in [plot_mario, plot_plant, plot_princess, plot_one_up, plot_jump, plot_tortoise]):
                    luigi = AnnotationBbox(OffsetImage(plt.imread('./images/luigi.png'), zoom=0.28),np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(luigi)
                
    
            # Plot for Luigi.
            elif plot_luigi and \
                    all(not item for item in [plot_mario, plot_plant, plot_princess, plot_one_up, plot_jump, plot_tortoise]):
                luigi = AnnotationBbox(OffsetImage(plt.imread('./images/luigi.png'), zoom=0.28),np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(luigi)
                
                if plot_mario and \
                        all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_one_up, plot_jump, plot_tortoise]):
                    mario = AnnotationBbox(OffsetImage(plt.imread('./images/mario.png'), zoom=0.28),np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(mario)


            # Plot for Plant.
            elif plot_plant and \
                    all(not item for item in [plot_mario,plot_luigi, plot_princess, plot_one_up, plot_jump, plot_tortoise]):
                plant = AnnotationBbox(OffsetImage(plt.imread('./images/plant.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(plant)

            # Plot for Princess.
            elif plot_princess and \
                    all(not item for item in [plot_mario,plot_luigi, plot_plant, plot_one_up, plot_jump, plot_tortoise]):
                princess = AnnotationBbox(OffsetImage(plt.imread('./images/princess.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(princess)

            # Plot for one_up.
            elif plot_one_up and \
                    all(not item for item in [plot_mario,plot_luigi, plot_plant, plot_princess, plot_jump, plot_tortoise]):
                one_up = AnnotationBbox(OffsetImage(plt.imread('./images/oneup.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(one_up)

            # Plot for Uphill.
            elif plot_jump and \
                    all(not item for item in [plot_mario,plot_luigi, plot_plant, plot_princess, plot_one_up, plot_tortoise]):
                jump = AnnotationBbox(OffsetImage(plt.imread('./images/jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(jump)

            # Plot for tortoise.
            elif plot_tortoise and \
                    all(not item for item in [plot_mario,plot_luigi, plot_plant, plot_princess, plot_one_up, plot_jump]):
                tortoise = AnnotationBbox(OffsetImage(plt.imread('./images/tortoise.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(tortoise)



            # Plot for mario and one_up.
            elif all(item for item in [plot_mario, plot_one_up]) and all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_jump, plot_tortoise]):
                mario_one_up = AnnotationBbox(OffsetImage(plt.imread('./images/mario_oneup.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(mario_one_up)
                
                # Plot for luigi and one_up.
                if all(item for item in [plot_luigi, plot_one_up]) and all(not item for item in [plot_mario, plot_plant, plot_princess, plot_jump, plot_tortoise]):
                    luigi_one_up = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_oneup.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(luigi_one_up)
                
        
            # Plot for luigi and one_up.
            elif all(item for item in [plot_luigi, plot_one_up]) and all(not item for item in [plot_mario, plot_plant, plot_princess, plot_jump, plot_tortoise]):
                luigi_one_up = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_oneup.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(luigi_one_up)
                
                if all(item for item in [plot_mario, plot_one_up]) and all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_jump, plot_tortoise]):
                    mario_one_up = AnnotationBbox(OffsetImage(plt.imread('./images/mario_oneup.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(mario_one_up)
                
           
                
             # Plot for mario and princess.
            elif all(item for item in [plot_mario, plot_princess]) and \
                    all(not item for item in [plot_luigi, plot_plant, plot_one_up, plot_jump, plot_tortoise]):
                mario_princess = AnnotationBbox(OffsetImage(plt.imread('./images/princess.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(mario_princess)
                exit()
                
                
             # Plot for luigi and princess.
            elif all(item for item in [plot_luigi, plot_princess]) and \
                    all(not item for item in [plot_mario, plot_plant, plot_one_up, plot_jump, plot_tortoise]):
                luigi_princess = AnnotationBbox(OffsetImage(plt.imread('./images/princess.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(luigi_princess)
                exit()
                
            # Plot for mario and plant.
            elif all(item for item in [plot_mario, plot_plant]) and \
                     all(not item for item in [plot_luigi, plot_princess, plot_one_up, plot_jump, plot_tortoise]):
                 mario_plant = AnnotationBbox(OffsetImage(plt.imread('./images/mario_plant.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                 ax.add_artist(mario_plant)
                 
                 # Plot for luigi and plant.
                 if all(item for item in [plot_luigi, plot_plant]) and \
                          all(not item for item in [plot_mario, plot_princess, plot_one_up, plot_jump, plot_tortoise]):
                      luigi_plant = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_plant.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                      ax.add_artist(luigi_plant)
                 
    
            # Plot for luigi and plant.
            elif all(item for item in [plot_luigi, plot_plant]) and \
                     all(not item for item in [plot_mario, plot_princess, plot_one_up, plot_jump, plot_tortoise]):
                 luigi_plant = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_plant.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                 ax.add_artist(luigi_plant)
                 
                 if all(item for item in [plot_mario, plot_plant]) and \
                          all(not item for item in [plot_luigi, plot_princess, plot_one_up, plot_jump, plot_tortoise]):
                      mario_plant = AnnotationBbox(OffsetImage(plt.imread('./images/mario_plant.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                      ax.add_artist(mario_plant)
                 

            # Plot for mario and jump.
            elif all(item for item in [plot_mario, plot_jump]) and \
                    all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_one_up, plot_tortoise]):
                mario_jump = AnnotationBbox(OffsetImage(plt.imread('./images/mario_jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(mario_jump)
                
                # Plot for luigi and jump.
                if all(item for item in [plot_luigi, plot_jump]) and \
                        all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_one_up, plot_tortoise]):
                    luigi_jump = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(luigi_jump)
                
            # Plot for luigi and jump.
            elif all(item for item in [plot_luigi, plot_jump]) and \
                    all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_one_up, plot_tortoise]):
                luigi_jump = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(luigi_jump)
                
                if all(item for item in [plot_mario, plot_jump]) and \
                        all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_one_up, plot_tortoise]):
                    mario_jump = AnnotationBbox(OffsetImage(plt.imread('./images/mario_jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(mario_jump)
                
                

            # Plot for mario, plant and jump.            
            elif all(item for item in [plot_mario, plot_plant, plot_jump]) and \
                    all(not item for item in [plot_luigi, plot_princess, plot_one_up, plot_tortoise]):
                mario_plant_jump = AnnotationBbox(OffsetImage(plt.imread('./images/mario_plant_jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(mario_plant_jump)
                
                if all(item for item in [plot_luigi, plot_plant, plot_jump]) and \
                        all(not item for item in [plot_mario, plot_princess, plot_one_up, plot_tortoise]):
                    luigi_plant_jump = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_plant_jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(luigi_plant_jump)
                
        
            # Plot for luigi, plant and jump.            
            elif all(item for item in [plot_luigi, plot_plant, plot_jump]) and \
                    all(not item for item in [plot_mario, plot_princess, plot_one_up, plot_tortoise]):
                luigi_plant_jump = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_plant_jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(luigi_plant_jump)
                
                if all(item for item in [plot_mario, plot_plant, plot_jump]) and \
                        all(not item for item in [plot_luigi, plot_princess, plot_one_up, plot_tortoise]):
                    mario_plant_jump = AnnotationBbox(OffsetImage(plt.imread('./images/mario_plant_jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(mario_plant_jump)
                

            # Plot for mario and tortoise.
            elif all(item for item in [plot_mario, plot_tortoise]) and \
                    all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_one_up, plot_jump]):
                mario_tortoise = AnnotationBbox(OffsetImage(plt.imread('./images/mario_tortoise.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(mario_tortoise)
                
                if all(item for item in [plot_luigi, plot_tortoise]) and \
                        all(not item for item in [plot_mario, plot_plant, plot_princess, plot_one_up, plot_jump]):
                    luigi_tortoise = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_tortoise.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(luigi_tortoise)
                
            # Plot for luigi and tortoise.
            elif all(item for item in [plot_luigi, plot_tortoise]) and \
                    all(not item for item in [plot_mario, plot_plant, plot_princess, plot_one_up, plot_jump]):
                luigi_tortoise = AnnotationBbox(OffsetImage(plt.imread('./images/luigi_tortoise.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(luigi_tortoise)
                
                if all(item for item in [plot_mario, plot_tortoise]) and \
                        all(not item for item in [plot_luigi, plot_plant, plot_princess, plot_one_up, plot_jump]):
                    mario_tortoise = AnnotationBbox(OffsetImage(plt.imread('./images/mario_tortoise.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                    ax.add_artist(mario_tortoise)
                

            #res needed
            # Plot for plant and jump.
            elif all(item for item in [plot_plant, plot_jump]) and \
                    all(not item for item in [plot_mario,plot_luigi, plot_princess, plot_one_up, plot_tortoise]):
                plant_jump = AnnotationBbox(OffsetImage(plt.imread('./images/plant_jump.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(plant_jump)
            
            
            else:
                nothing = AnnotationBbox(OffsetImage(plt.imread('./images/blue_sky.png'), zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(nothing)
                

           

        coordinates_state_mapping_2 = {}
        for j in range(self.env_height * self.env_width):
            coordinates_state_mapping_2[j] = np.asarray(
                [j % self.env_width, int(np.floor(j / self.env_width))])
            
        for position in coordinates_state_mapping_2:
            plot_image(coordinates_state_mapping_2[position])

        #plotting the coordinates on x and y axis
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        plt.grid()  

        #calling plt.show method to render images at each step of agent, plot - true - passed in arguments
        if plot:  
            plt.show()
        else:
            #plotting the canvas by choosing the size as 90x90
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            width = 90
            height = 90
            dim = (width, height)
            preprocessed_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            plt.show()
            return preprocessed_image


def print_mario_QTable():
    print("Mario's Q table")
    print(Qtable_mario)

def print_luigi_QTable():
    print("Luigi's Q table")
    print(Qtable_luigi)

def randomstep():
    list1 = [0, 1, 2, 3]
    for i in range(0,steps):
        #print("running")
        res_mario = random.choice(list1)
        res_luigi = random.choice(list1)
        return res_mario, res_luigi


#Q learning for mario and luigi
def SARSA_training_algorithm(episodes):
    
    #----------------------SARSA TRAINING------------------------
    
    mariobros = MariosBros(env_type='determninistic')
    mariobros.reset()
    for i in range(0,episodes):
        done_mario = False
        done_luigi = False
        
        epsilon_new_val = EpsilonDecay()
        epsilonDecayArray.append(epsilon_new_val)
        
        for j in range(0,40):
            action_mario , action_luigi = randomstep()
            observation_mario ,observation_luigi, reward_mario, reward_luigi, done_mario, done_luigi, info = (mariobros.step(action_mario, action_luigi))
            print(observation_mario ,observation_luigi, reward_mario, reward_luigi, done_mario, done_luigi, info)
            
            mario_rewardArray.append(reward_mario)
            luigi_rewardArray.append(reward_luigi)
            
            x_cord_mario , y_cord_mario = Get_cord_from_pos(observation_mario)
            x_cord_luigi , y_cord_luigi = Get_cord_from_pos(observation_luigi)
            
            #print("coordinates for mario are:",x_cord_mario," and ", y_cord_mario)
            #print("coordinates for luigi are:",x_cord_luigi," and ", y_cord_luigi)
            
            #for mario
            Qnew_mario , state_val_mario , action_mario = Get_Q_learning_value(reward_mario, observation_mario, Qtable_mario , x_cord_mario, y_cord_mario, action_mario)
            Qtable_mario[state_val_mario][action_mario] = Qnew_mario
            
            #for luigi
            Qnew_luigi , state_val_luigi , action_luigi = Get_Q_learning_value(reward_mario, observation_mario, Qtable_mario , x_cord_mario, y_cord_mario, action_mario)
            Qtable_luigi[state_val_luigi][action_luigi] = Qnew_luigi
            
            if(done_mario or done_luigi):
                break
            

#Helper function for getting Q values
def Get_Q_learning_value(initialreward, observation, Qtable, x_cord, y_cord, action):
    # Q learning code
    #Formula  - Qnew = Qold + alpha(R + gamma(Qold(max val)) - Qold)
    
    state_val =  Map_state_Key_Value(x_cord,y_cord)
    Qold = Qtable [state_val] [action]
    num = Qtable [state_val]
    
    Qmax = max(num)
    Qnew = 0
    #print("max in row:",num , " is",Qmax)
    Qnew = Qold + alpha *(initialreward + (gamma * Qmax) - Qold)
    #print("new Q value is:",Qnew)
    return Qnew, state_val, action

def digit_extraction_by_index(x, n):
    return (abs(x) // (10 ** n)) % 10    

def Get_cord_from_pos(carpos):
     if(carpos==0):
         return 0,0
     elif(carpos==1):
         return 0,1
     elif(carpos==2):
         return 0,2
     elif(carpos==3):
         return 0,3
     elif(carpos==4):
         return 0,4
     elif(carpos==5):
         return 0,5
     elif(carpos==6):
         return 0,6
     elif(carpos==7):
         return 0,7
     elif(carpos==8):
         return 0,8
     elif(carpos==9):
         return 0,9
     elif(carpos==10):
         return 1,0
     #print("carpos is",carpos)
     xcord = digit_extraction_by_index(carpos, 1)
     ycord = digit_extraction_by_index(carpos, 2)
     #print("xcord",xcord," ycord",ycord)
     return xcord,ycord
 
def getAppropiate(arr):
    if(arr[0] == 0):
        return 0
    elif(arr[1] == 0):
        return 1
    elif(arr[2]==0):
        return 2
    elif(arr[3]==0):
        return 3;
    else:
        return 0 #since all 4 column value must be same

def Map_state_Key_Value(row, col):
    return stateMap.get((row,col))

def EpsilonDecay():
    n = epsilonDecayArray[-1]
    n = n * epsilon
    epsilonDecayArray.append(n)
    return n    



#defining the state matrix
stateMap = { (0,0):0, (0,1):1, (0,2):2, (0,3):3, (0,4):4, (0,5):5, (0,6):6, (0,7):7, (0,8):9, (0,9):10 ,  
            (1,0):11, (1,1):12, (1,2):13, (1,3):14, (1,4):15, (1,5):16, (1,6):17, (1,7):18, (1,8):19, (1,9):20,
            (2,0):21, (2,1):22, (2,2):23, (2,3):24, (2,4):25, (2,5):26, (2,6):27, (2,7):28, (2,8):29, (2,9):30, 
            (3,0):31, (3,1):32, (3,2):33, (3,3):34, (3,4):35, (3,5):36, (3,6):37, (3,7):38, (3,8):39, (3,9):40, 
            (4,0):41, (4,1):42, (4,2):43, (4,3):44, (4,4):45, (4,5):46, (4,6):47, (4,7):48, (4,8):49, (4,9):50, 
            (5,0):51, (5,1):52, (5,2):53, (5,3):54, (5,4):55, (5,5):56, (5,6):57, (5,7):58, (5,8):59, (5,9):60, 
            (6,0):61, (6,1):62, (6,2):63, (6,3):64, (6,4):65, (6,5):66, (6,6):67, (6,7):68, (6,8):69, (6,9):70,
            (7,0):71, (7,1):72, (7,2):73, (7,3):74, (7,4):75, (7,5):76, (7,6):77, (7,7):78, (7,8):79, (7,9):80, 
            (8,0):81, (8,1):82, (8,2):83, (8,3):84, (8,4):85, (8,5):86, (8,6):87, (8,7):88, (8,8):89, (8,9):90, 
            (9,0):91, (9,1):92, (9,2):93, (9,3):94, (9,4):95, (9,5):96, (9,6):97, (9,7):98, (9,8):99, (9,9):100
            }



#defining the hyper-parameters for Q learning algorithm
states  = 100 # we have 10*10 grid
rows = 10
cols = 10
actions = 4 # agent can perform 4 actions left,right,up,down
alpha = 0.2 
gamma = 0.8
epsilon = 0.95 
epochs = 10000
steps = 40 #by default in each episode , the agent will try to run for 40 steps

Qtable_mario = np.zeros((100, 4))
Qtable_luigi = np.zeros((100, 4))

epsilonDecayArray = []

epsilonDecayArray.append(0.98)

mario_rewardArray = []
luigi_rewardArray = []
    
#main driver logic
env_num = int(input('Please select type of environment: 1-deterministic and 2-stochastic 3-Sarsa-multiagent:'))
print("Environment chosen is:",env_num)



if env_num == 1:
    #code for deterministic environment
    steps = int(input('Please enter the number of steps you want to take :'))
    print("The agent will take",steps,"steps now")

    mariobros = MariosBros(env_type='deterministic')
    mariobros.reset()
    # creating a list of actions - up,down,right,left - each has 0.25 probability
    list1 = [0, 1, 2, 3]
    for i in range(0,steps):
        #print("running")
        res_mario = random.choice(list1)
        res_luigi = random.choice(list1)
        #print("random choice by deterministic env is: ", res)
        print(mariobros.step(res_mario,res_luigi))
        mariobros.render(plot=True)
elif env_num == 2:
    #code for stochastic environment
    mariobros = MariosBros(env_type='stochastic')
    steps = int(input('Please enter the number of steps you want to take :'))
    print("The agent will take",steps,"steps now")

    mariobros.reset()
    for i in range(0,steps):
        print("running")
        step_to_take = (random.randint(1, 100))
        if(step_to_take<90):
            if(random.choice([1, 2]) == 1):
                action_mario = 0
                action_luigi = 1
                #print("stochastic choice by env is: ", action)
                print(mariobros.step(action_mario, action_luigi))
                mariobros.render(plot=True)
            else:
                action_mario = 2
                action_luigi = 2
                #print("stochastic choice by env is: ", action)
                print(mariobros.step(action_mario, action_luigi))
                mariobros.render(plot=True)
        else:
            if(random.choice([1, 2]) == 1):
                action_mario = 1
                action_luigi = 0
                #print("stochastic choice by env is: ", action)
                print(mariobros.step(action_mario, action_luigi))
                mariobros.render(plot=True)
            else:
                action_mario = 3
                action_luigi = 3
                #print("stochastic choice by env is: ", action)
                print(mariobros.step(action_mario, action_luigi))
                mariobros.render(plot=True)
elif env_num==3:
    episodes = int(input('Please enter the number of episodes:'))
    print("Running the agent for ",episodes," now")

    SARSA_training_algorithm(episodes)
    print_mario_QTable()
    print_luigi_QTable()
else:
    print('Invalid choice')


print("END OF EXECUTION") 

#references taken - Nitin Kulkarni's code uploaded on Piazza for deciding the movement of the agent
#modified all other parts like grid length and grid size and the proportions to the image






        

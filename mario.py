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


#main driver logic
env_num = int(input('Please select type of environment: 1-deterministic and 2-stochastic :'))
print("Environment chosen is:",env_num)

steps = int(input('Please enter the number of steps you want to take :'))
print("The agent will take",steps,"steps now")


if env_num == 1:
    #code for deterministic environment
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
    mariobros.reset()
    for i in range(0,steps):
        print("running")
        step_to_take = (random.randint(1, 100))
        if(step_to_take<90):
            if(random.choice([1, 2]) == 1):
                action = 0
                #print("stochastic choice by env is: ", action)
                print(mariobros.step(action))
                mariobros.render(plot=True)
            else:
                action = 2
                #print("stochastic choice by env is: ", action)
                print(mariobros.step(action))
                mariobros.render(plot=True)
        else:
            if(random.choice([1, 2]) == 1):
                action = 1
                #print("stochastic choice by env is: ", action)
                print(mariobros.step(action))
                mariobros.render(plot=True)
            else:
                action = 3
                #print("stochastic choice by env is: ", action)
                print(mariobros.step(action))
                mariobros.render(plot=True)
else:
    print('Invalid choice')


    
    

print("END OF EXECUTION") 

#references taken - Nitin Kulkarni's code uploaded on Piazza for deciding the movement of the agent
#modified all other parts like grid length and grid size and the proportions to the image






        

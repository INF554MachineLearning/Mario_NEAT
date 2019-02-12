import retro
import numpy as np
import neat
import pickle
import cv2

#env = retro.make('SpaceInvaders-Nes', '1Player.Level1')
env = retro.make('SuperMarioBros-Nes', 'Level1-1')
#env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
#env = retro.make('SuperMarioWorld-Snes', 'Start')

imgarray = []
attention = np.array([
    [0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1],
    [0.75, 0.7, 0.65, 0.55, 0.45, 0.35, 0.25],
    [0.85, 0.8, 0.75, 0.65, 0.55, 0.45, 0.35],
    [0.95, 0.9, 0.85, 0.75, 0.65, 0.55, 0.45],
    [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    [1.0, 0.95, 0.9, 0.85, 0.75, 0.65, 0.55],
    [0.95, 0.9, 0.85, 0.75, 0.65, 0.55, 0.45]
    ])

xpos_end = 0


def eval_genomes(genomes, config):


    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0
        page1 = np.zeros((13,16))
        page2 = np.zeros((13,16))
        pages = [page1, page2]
        info = {}
        done = False
        player_x = 0
        player_y = 10
        page_cur = 0
        enemies = np.array([[0, 0, 0, 0]]*5)
        player_yspeed = 0

        while not done:
            env.render()
            frame += 1
            

            if frame>1:
                px_before = player_x
                player_x = max(0, int(info['player_x']/16))
                if px_before > player_x:
                    page_cur = 1-page_cur
                player_y = max(7, min(int(info['player_y']/16), 11))
                player_yspeed = info['player_yspeed']
                if player_yspeed < 0: player_yspeed=1
                else: player_yspeed = 0

                for i in range(13):
                    for j in range(16):
                        name1 = "p1_"+str(i)+","+str(j)
                        name2 = "p2_"+str(i)+","+str(j)
                        page1[i][j] = min(1,info[name1])
                        page2[i][j] = min(1,info[name2])

                for i in range(5):
                    enemies[i,0] = int(info["enemy"+str(i)+"_drawn"])
                    enemies[i,1] = int(info["enemy"+str(i)+"_x"]/16)
                    enemies[i,2] = min(12,max(0,int(info["enemy"+str(i)+"_y"]/16)-1))
                    enemies[i,3] = int(info["enemy"+str(i)+"_sx"])%2
                    if enemies[i,0] == 1:
                        pages[enemies[i,3]][enemies[i,2],enemies[i,1]]=-1
                #print(player_yspeed)

            combined_pages = np.column_stack([pages[page_cur], pages[1-page_cur]])
            feed = combined_pages[player_y-2:player_y+2, player_x:player_x+4]
            block = 1
            enemy_ahead = 1
            enemy_below = 1
            pit = 0
            if feed[1,2]==0 and feed[1,1]==0:
                block=0
            if feed[1,2]!=-1 and feed[1,1]!=-1 and feed[1,3]!=-1:
                enemy_ahead=0
            if feed[2,1]!=-1 and feed[2,2]!=-1 and feed[2,3]!=-1 and feed[3,1]!=-1 and feed[3,2]!=-1 and feed[3,3]!=-1:
                enemy_below=0
            if feed[2,1]==0 and feed[3,1]==0:
                pit = 1
            #print(player_x, player_y)
            #cv2.imshow('main', feed)
            #cv2.waitKey(1)
            imgarray = np.ndarray.flatten(feed)
            #cv2.imshow('main', feed)
            #cv2.waitKey(1) 

            [right, jump] = net.activate([player_yspeed, block, enemy_ahead, enemy_below, pit])
            if right>0.5: right = 1
            else: right = 0
            if jump>0.5: jump = 1
            else: jump = 0
            buttons = [0]*12
            buttons[7]=right
            buttons[8]=jump
            ob, rew, done, info = env.step(buttons)
            
            #xpos = info['x']
            #xpos_end = info['screen_x_end']
            
            
            #if xpos > xpos_max:
                #fitness_current += 1
                #xpos_max = xpos
            
            #if xpos == xpos_end and xpos > 500:
                #fitness_current += 100000
                #done = True
            
            fitness_current += rew
            
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                
            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)
                
            genome.fitness = fitness_current
                
            
            
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))
#p = neat.Checkpointer().restore_checkpoint('neat-checkpoint-448')
winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
    


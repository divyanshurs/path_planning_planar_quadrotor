import numpy as np


class GridWorld():


  ''' 
  Gridworld dynamics very loosely borrowed from Sutton and Barto:
  Reinforcement Learning

  The dynamics have been simplified to be deterministic
  '''
  def __init__(self, slip_prob = 0.1):
    # actions legend 
    # 0: N 
    # 1: E 
    # 2: S 
    # 3: W 
    self.rows = 10
    self.cols = 10

    base_cost = 1
    large_cost = 5
    wall_cost = 2

    slow_states = [1 * self.rows + 2, 2 * self.rows + 2, 
                   3 * self.rows + 2, 4 * self.rows + 2,
                   5 * self.rows + 4, 5 * self.rows + 5, 
                   5 * self.rows + 6, 5 * self.rows + 7, 
                   5 * self.rows + 8]
    goal_state = 2 * self.rows + 4
    # print(slow_states)        
    P = np.zeros((self.rows * self.cols,4), dtype=(list))

    # TODO: Fill out the transition matrix P
    # P is a n x m matrix whose elements are a list of 2 tuples, the contents of 
    # each tuple is (probability, next_state, cost)
    #
    # The entry for the goal state has already been completed for you. You may 
    # find the convenience functions map_row_col_to_state() and 
    # map_state_to_row_col() helpful but are not required to use them.

    # for s in range(self.rows * self.cols):
    #   if(s == goal_state):
    #     # Taking any action at the goal state will stay at the goal state
    #     P[s][0] = [(1.0, s, 0),
    #                (0.0, 0, 0)]
    #     P[s][1] = [(1.0, s, 0),
    #                (0.0, 0, 0)]
    #     P[s][2] = [(1.0, s, 0),
    #                (0.0, 0, 0)]
    #     P[s][3] = [(1.0, s, 0),
    #                (0.0, 0, 0)]
    #     break

    # actions legend 
    # 0: N 
    # 1: E 
    # 2: S 
    # 3: W 
      # STUDENT CODE HERE
    slow_states_up_before = np.array(slow_states)[:4] -1
    slow_states_up_after = np.array(slow_states)[:4] +1
    slow_state_down = np.array(slow_states)[4:]
    row_col_down = []
    for w in slow_state_down:
        row_col_down.append(self.map_state_to_row_col(w))
    
    row_col_down_up = row_col_down_down = np.array(row_col_down)
    # print(row_col_down_up)
    row_col_down_up[:,0] = row_col_down_up[:,0] - 1 
    # print(row_col_down_up)

    row_col_down_down[:,0] = row_col_down_down[:,0] + 1 
    # print(row_col_down)
    # print(slow_state_down_up)
    # print(slow_states_up_before)
    for row in range(self.rows):
        for col in range(self.cols):
            s = self.map_row_col_to_state(row,col)
            P[s][0] = [(1-slip_prob, self.map_row_col_to_state(row-1,col), base_cost),
                        (slip_prob, s, base_cost)]
            P[s][1] = [(1-slip_prob, self.map_row_col_to_state(row,col+1), base_cost),
                        (slip_prob, s, base_cost)] 
            P[s][2] = [(1-slip_prob, self.map_row_col_to_state(row+1,col), base_cost),
                        (slip_prob, s, base_cost)]
            P[s][3] = [(1-slip_prob, self.map_row_col_to_state(row,col-1), base_cost),
                        (slip_prob, s, base_cost)]
            if(row==0): #first row
                P[s][0] = [(1-slip_prob, s, wall_cost),
                            (slip_prob, s, base_cost)]
                if(col==0): #first cell
                    P[s][3] = [(1-slip_prob, s, wall_cost),
                                (slip_prob, s, base_cost)]
                if(col==self.cols-1): #last cell
                    P[s][1] = [(1-slip_prob, s, wall_cost),
                                (slip_prob, s, base_cost)] 
                # if(s==self.map_row_col_to_state(0,2)): #special obs cell
                #     P[s][2] = [(1-slip_prob, self.map_row_col_to_state(row+1,col), large_cost),
                #             (slip_prob, s, base_cost)]
            if(row==self.rows-1): #last row
                P[s][2] = [(1-slip_prob, s, wall_cost),
                            (slip_prob, s, base_cost)]
                if(col==0): #bottom first cell
                    P[s][3] = [(1-slip_prob, s, wall_cost),
                                (slip_prob, s, base_cost)]
                if(col==self.cols-1): #last cell
                    P[s][1] = [(1-slip_prob, s, wall_cost),
                                (slip_prob, s, base_cost)] 
            if(col==0 and row!=0 and row!=self.rows-1): #first col excluding rows
                P[s][3] = [(1-slip_prob, s, wall_cost),
                            (slip_prob, s, base_cost)]    
            if(col==self.cols-1 and row!=0 and row!=self.rows-1): #last cols excluding cols
                P[s][1] = [(1-slip_prob, s, wall_cost),
                            (slip_prob, s, base_cost)]
                # if(s==self.map_row_col_to_state(5,9)): #special obs cell
                #     P[s][3] = [(1-slip_prob, self.map_row_col_to_state(row,col-1), large_cost),
                #         (slip_prob, s, base_cost)]
            # print((np.array(s) == slow_states).all())
            for kk in slow_states:
                if(s==kk):
            # if(np.any(s == slow_states)): #if in slow state
                    # print(s)
                    P[s][0] = [(1-slip_prob, self.map_row_col_to_state(row-1,col), large_cost),
                                (slip_prob, s, large_cost)]
                    P[s][1] = [(1-slip_prob, self.map_row_col_to_state(row,col+1), large_cost),
                                (slip_prob, s, large_cost)] 
                    P[s][2] = [(1-slip_prob, self.map_row_col_to_state(row+1,col), large_cost),
                                (slip_prob, s, large_cost)]
                    P[s][3] = [(1-slip_prob, self.map_row_col_to_state(row,col-1), large_cost),
                                (slip_prob, s, large_cost)]
                    break
            # for jj in slow_states_up_before:
            #     if(s==jj):
            # # if(np.any(s==slow_states_up_before)):#up slow state on the left side
            #         P[s][1] = [(1-slip_prob, self.map_row_col_to_state(row,col+1), large_cost),
            #             (slip_prob, s, base_cost)]
            #         break
            
            # for ll in slow_states_up_after:
            #     if(s==ll):
            # # if(np.any(s==slow_states_up_after)):#up slow state on the right side
            #         P[s][3] = [(1-slip_prob, self.map_row_col_to_state(row,col-1), large_cost),
            #             (slip_prob, s, base_cost)]
            #         break

            # if(s==self.map_row_col_to_state(5,2)): #special one cell for up obstacle
            #     P[s][0] = [(1-slip_prob, self.map_row_col_to_state(row-1,col), large_cost),
            #             (slip_prob, s, base_cost)]
            
            # for mm in row_col_down_up[:,0]:
            #     for tt in row_col_down_up[:,1]:
            #         if(row==mm and col == tt):


            # # if(np.any(row == row_col_down_up[:,0]) and np.any(col== row_col_down_up[:,1])): #down obs up
            #             P[s][2] = [(1-slip_prob, self.map_row_col_to_state(row+1,col), large_cost),
            #                 (slip_prob, s, base_cost)]
            #             break
            
            # for gg in row_col_down_down[:,0]:
            #     for hh in row_col_down_down[:,1]:
            #         if(row==gg and col == hh):


            # # if(np.any(row == row_col_down_down[:,0]) and np.any(col== row_col_down_down[:,1])): #down obs down
            #             P[s][0] = [(1-slip_prob, self.map_row_col_to_state(row-1,col), large_cost),
            #                 (slip_prob, s, base_cost)]
            #             break
            
            # if(s == self.map_row_col_to_state(5,3)): #special case for down obs
            #     P[s][1] = [(1-slip_prob, self.map_row_col_to_state(row,col+1), large_cost),
            #             (slip_prob, s, base_cost)]
                        
            if(s == goal_state):
                # Taking any action at the goal state will stay at the goal state
                P[s][0] = [(1.0, s, 0),
                        (0.0, 0, 0)]
                P[s][1] = [(1.0, s, 0),
                        (0.0, 0, 0)]
                P[s][2] = [(1.0, s, 0),
                        (0.0, 0, 0)]
                P[s][3] = [(1.0, s, 0),
                        (0.0, 0, 0)]

    self.P = P

  def map_row_col_to_state(self, row, col):
    return row * self.rows + col

  def map_state_to_row_col(self, state):
    return state // self.cols, np.mod(state, self.cols)

  def eval_action(self, state, action):
    row, col = self.map_state_to_row_col(state)
    if action < 0 or action > 3:
      raise ValueError('Not a valid action')
    if row < 0 or row >= self.rows:
      raise ValueError('Row out of bounds')
    if col < 0 or col >= self.cols:
      raise ValueError('Col out of bounds')
    return self.P[state, action]

# g = GridWorld()
# print(g.P[2])
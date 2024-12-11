import numpy as np
input_array = np.array([1, 3, -1, 3, 1, 5])
# print(input_array)
root_node = input_array[0]
# print("root node:", root_node)
layer2_nodes = input_array[1:3]
# print("layer2 nodes:", layer2_nodes)
goal_nodes = input_array[3:]
# print("goal nodes:", goal_nodes)
print("root node:", root_node)
print("layer2 nodes:", layer2_nodes)
print("goal nodes:", goal_nodes)
blue_path =root_node + layer2_nodes[0] +goal_nodes[0]
print("blue path:", blue_path)
red_path = root_node + layer2_nodes[0] + goal_nodes[1]
print("red path:", red_path)
green_path = root_node + layer2_nodes[1] + goal_nodes[2]
print("green path:", green_path)
black_path = root_node + layer2_nodes[1] + goal_nodes[1]
print("black path:", black_path)

path = {
    'blue': blue_path,
    'black': black_path,
    'green': green_path,
    'red': red_path}
# print(path.items())
# print(path.values())
# print(path.keys())
# print(max(path.items(), key=lambda x: x[1]))
# print(max(path.items(), key=lambda x: x[0]))
name, the_highest_value_path = max(path.items(), key=lambda x: x[1])
# print(the_highest_value_path,name)
if name == "blue":
    print(f"The highest value path is blue path and the value is {the_highest_value_path} and the path is {root_node},{layer2_nodes[0]},{goal_nodes[0]}")
elif name == "red":
    print(f"The highest value path is red path and the value is {the_highest_value_path} and the path is {root_node},{layer2_nodes[0]},{goal_nodes[1]}")
elif name == "green":
    print(f"The highest value path is green path and the value is {the_highest_value_path} and the path is {root_node},{layer2_nodes[1]},{goal_nodes[2]}")
elif name == "black":
    print(f"The highest value path is black path and the value is {the_highest_value_path} and the path is {root_node},{layer2_nodes[1]},{goal_nodes[1]}")
import ch03.trees as trees
import ch03.tree_plotter as tree_plotter

fr = open("lenses.txt")
lenses = [inst.strip().split("\t") for inst in fr.readlines()]
lenses_labels = ["age", "prescript", "astigmatic", "tearRate"]
lenses_tree = trees.create_tree(lenses,lenses_labels)
print(lenses_tree)
tree_plotter.create_plot(lenses_tree)

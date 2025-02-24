import random
import argparse
from collections import defaultdict

MAX_PARTICLES_PER_CELL = 128

def generate_pdb(filename, num_particles, box_size, cell_cutoff_radius):
    # need to check for unique positions
    positions = set() 
    cell_counts = defaultdict(int)

    cell_size = cell_cutoff_radius
    num_cells_per_dim = int(box_size/cell_size)

    with open(filename, "w") as f:
        while len(positions) < num_particles:
            x = round(random.uniform(0, box_size), 3)
            y = round(random.uniform(0, box_size), 3)
            z = round(random.uniform(0, box_size), 3)
            cell_x = int(x/cell_size)
            cell_y = int(y/cell_size)
            cell_z = int(z/cell_size)
            cell_index = (cell_x, cell_y, cell_z)

            if cell_counts[cell_index] < MAX_PARTICLES_PER_CELL:
                position = (x,y,z)
                if position not in positions:
                    positions.add(position)
                    cell_counts[cell_index] += 1
                    atom_line = f"ATOM  {len(positions):5d}  X   XXX A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           X  \n"
                    f.write(atom_line)
        f.write("END\n")
    print(f"PDB file '{filename}' created with {num_particles} particles in a {box_size} Angstrom^3 box.")

def main():
    parser = argparse.ArgumentParser(description="generate PDB file with unique particles in random positions")
    parser.add_argument("-o", "--output", type=str, required=True, help="output PDB filename")
    parser.add_argument("-n", "--num_particles", type=int, required=True, help="number of particles")
    parser.add_argument("-b", "--box_size", type=int, required=True, help="size of cube defined by one dimension in Angstroms")
    parser.add_argument("-c", "--cell_cutoff_radius", type=int, required=True, help="size of cube defined by one dimension in Angstroms")
    

    args = parser.parse_args()
    generate_pdb(args.output, args.num_particles, args.box_size, args.cell_cutoff_radius)


if __name__ == '__main__':
    main()
import random
import argparse

def generate_pdb(filename, num_particles, box_size):
    # need to check for unique positions
    positions = set() 

    with open(filename, "w") as f:
        while len(positions) < num_particles:
            x = round(random.uniform(0, box_size), 3)
            y = round(random.uniform(0, box_size), 3)
            z = round(random.uniform(0, box_size), 3)
            position = (x,y,z)
            if position not in positions:
                positions.add(position)
                atom_line = f"ATOM  {len(positions):5d}  X   XXX A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           X  \n"
                f.write(atom_line)
        f.write("END\n")
    print(f"PDB file '{filename}' created with {num_particles} particles in a {box_size} Angstrom^3 box.")

def main():
    parser = argparse.ArgumentParser(description="generate PDB file with unique particles in random positions")
    parser.add_argument("-o", "--output", type=str, required=True, help="output PDB filename")
    parser.add_argument("-n", "--num_particles", type=int, required=True, help="number of particles")
    parser.add_argument("-b", "--box_size", type=int, required=True, help="size of cube defined by one dimension in Angstroms")

    args = parser.parse_args()
    generate_pdb(args.output, args.num_particles, args.box_size)


if __name__ == '__main__':
    main()
import os
import click
import pickle
from random import randint
from pxblat import Server, Client

@click.command(name="Blat")
@click.option("-i", "--input",
              type=click.Path(exists=True, file_okay=True),
              metavar="FILE",
              required = True,
              help="List of sequences (one per row)")
@click.option("-g", "--genomeDir", "genomeDir",
              type=click.Path(exists=True, file_okay=False),
              metavar="DIR",
              required = True,
              help="Directory containing the genome files required for Blat (2.bit)")
@click.option("-o", "--output",
              type=click.STRING,
              metavar="NAME",
              required = True,
              help="Output name for the pickle")
def Blat(input, genomeDir, output):

    """
    pxBlat command to run many sequences at the same time
    """

    # File with sequences
    with open(input, 'r') as f:
        sequences:list = f.readlines()
    sequences = [line.rstrip('\n') for line in sequences]

    # 2bit file
    all_files:list = os.listdir(genomeDir)
    g2bit:click.Path = [file for file in all_files if file.endswith(".2bit")][0]

    # Blat options
    port:int = randint(6000, 7000)
    ## Client
    client = Client(
        host="localhost",
        port=port,
        seq_dir=genomeDir,
        min_score=20,
        min_identity=90
    )

    ## Server
    with Server("localhost", port, os.path.join(genomeDir, g2bit), can_stop=True, step_size=5, tile_size=10) as server: #BLAT WEB options
        server.wait_ready()  
        results = client.query(sequences)
    
    # Save results
    with open(output, 'wb') as f:
        pickle.dump(results, f)
    
    click.echo(f"Blat finished for {input}")

if __name__ == '__main__':
    Blat()

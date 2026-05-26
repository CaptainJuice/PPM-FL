import numpy as np
import random

def read_off(file):
    line = file.readline().strip()
    if not line:
        raise Exception('Empty file')
    if line.startswith('OFF'):
        if len(line) > 3:
            # Some files have header like "OFF 123 456 0"
            parts = line[3:].split()
            if len(parts) >= 2:
                n_verts, n_faces = int(parts[0]), int(parts[1])
                verts = [[float(s) for s in file.readline().strip().split()] for i_vert in range(n_verts)]
                faces = [[int(s) for s in file.readline().strip().split()][1:] for i_face in range(n_faces)]
                return verts, faces
        # Standard OFF header on its own line
        line = file.readline().strip()
    
    # Handle files where 'OFF' and counts are separated, or counts are on the second line
    while not line or line.startswith('#'):
        line = file.readline().strip()
        
    parts = line.split()
    n_verts, n_faces = int(parts[0]), int(parts[1])
    verts = [[float(s) for s in file.readline().strip().split()] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split()][1:] for i_face in range(n_faces)]
    return verts, faces

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        return sampled_points

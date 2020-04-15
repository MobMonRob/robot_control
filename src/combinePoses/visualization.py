import matplotlib.pyplot as plt
import numpy as np
import pylab
import matplotlib
#matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

default_axis_limits = {"x": [-400,750], "y": [-400,750], "z": [0,2000]}


#def visualize(poses, reference_pose):
#    global ax, quiver, line
#    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
#
#    reference_cs = common.coordinate_system_for_affine(reference_pose.to_affine())
#    reference_quiver = ax.quiver(*reference_cs.transpose(), colors=common.base_cs_colors, arrow_length_ratio=0.0)
#    pose_css = [common.coordinate_system_for_affine(pose.to_affine()) for pose in poses]
#    quivers = [ax.quiver(*cs.transpose(), colors=common.cs_colors, arrow_length_ratio=0.0, length=0.05) for cs in pose_css]

#    position_vectors = []
#    for pose in poses:
#        start_pos = reference_cs[:,:3]
#        direction = pose.to_affine()[:3,3] - start_pos
#        position_vectors.append(np.hstack([start_pos, direction]))
#    lines = [ax.quiver(*np.transpose(position_vectors), arrow_length_ratio=0.00, linewidths=0.05)]
#    ax.set_xlim(-1.5, 1.5)
#    ax.set_ylim(-1.5, 1.5)
#    ax.set_zlim(-1.5, 1.5)
#    ax.set_proj_type("persp")
#
#    plt.show()

def createPosSeries(posesDict):
    poseSeries = [[], [], [], [], [], [], [], [], [], []]
    for poseDict in posesDict:
        poseSeries[0].append([poseDict["LeftForearm"]["tx"], poseDict["LeftForearm"]["ty"], poseDict["LeftForearm"]["tz"]])
        poseSeries[1].append([poseDict["LeftElbow"]["tx"], poseDict["LeftElbow"]["ty"], poseDict["LeftElbow"]["tz"]] )
        poseSeries[2].append([poseDict["LeftHumerus"]["tx"], poseDict["LeftHumerus"]["ty"], poseDict["LeftHumerus"]["tz"]])
        poseSeries[3].append([poseDict["LeftScapula"]["tx"], poseDict["LeftScapula"]["ty"], poseDict["LeftScapula"]["tz"]])
        poseSeries[4].append([poseDict["ThoraxRaphael"]["tx"], poseDict["ThoraxRaphael"]["ty"], poseDict["ThoraxRaphael"]["tz"]])
        poseSeries[5].append([poseDict["RightScapula"]["tx"], poseDict["RightScapula"]["ty"], poseDict["RightScapula"]["tz"]])
        poseSeries[6].append([poseDict["RightHumerus"]["tx"], poseDict["RightHumerus"]["ty"], poseDict["RightHumerus"]["tz"]])
        poseSeries[7].append([poseDict["RightElbow"]["tx"], poseDict["RightElbow"]["ty"], poseDict["RightElbow"]["tz"]])
        poseSeries[8].append([poseDict["RightForearm"]["tx"], poseDict["RightForearm"]["ty"], poseDict["RightForearm"]["tz"]])
        poseSeries[9].append([(poseDict["LeftScapula"]["tx"]+poseDict["RightScapula"]["tx"])/2, (poseDict["LeftScapula"]["ty"]+poseDict["RightScapula"]["ty"])/2, (poseDict["LeftScapula"]["tz"]+poseDict["RightScapula"]["tz"])/2])
    return poseSeries

def createPoseLists(posesDict):
    poseSeries = [[[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]]
    for poseDict in posesDict:
        poseSeries[0][0].append(poseDict["LeftForearm"]["tx"])
        poseSeries[0][1].append(poseDict["LeftForearm"]["ty"])
        poseSeries[0][2].append(poseDict["LeftForearm"]["tz"])
        poseSeries[1][0].append(poseDict["LeftElbow"]["tx"])
        poseSeries[1][1].append(poseDict["LeftElbow"]["ty"] )
        poseSeries[1][2].append(poseDict["LeftElbow"]["tz"] )
        poseSeries[2][0].append(poseDict["LeftHumerus"]["tx"])
        poseSeries[2][1].append(poseDict["LeftHumerus"]["ty"])
        poseSeries[2][2].append(poseDict["LeftHumerus"]["tz"])
        poseSeries[3][0].append(poseDict["LeftScapula"]["tx"])
        poseSeries[3][1].append(poseDict["LeftScapula"]["ty"])
        poseSeries[3][2].append(poseDict["LeftScapula"]["tz"])
        poseSeries[4][0].append(poseDict["ThoraxRaphael"]["tx"])
        poseSeries[4][1].append(poseDict["ThoraxRaphael"]["ty"])
        poseSeries[4][2].append(poseDict["ThoraxRaphael"]["tz"])
        poseSeries[5][0].append(poseDict["RightScapula"]["tx"])
        poseSeries[5][1].append(poseDict["RightScapula"]["ty"])
        poseSeries[5][2].append(poseDict["RightScapula"]["tz"])
        poseSeries[6][0].append(poseDict["RightHumerus"]["tx"])
        poseSeries[6][1].append(poseDict["RightHumerus"]["ty"])
        poseSeries[6][2].append(poseDict["RightHumerus"]["tz"])
        poseSeries[7][0].append(poseDict["RightElbow"]["tx"])
        poseSeries[7][1].append(poseDict["RightElbow"]["ty"])
        poseSeries[7][2].append(poseDict["RightElbow"]["tz"])
        poseSeries[8][0].append(poseDict["RightForearm"]["tx"])
        poseSeries[8][1].append(poseDict["RightForearm"]["ty"])
        poseSeries[8][2].append(poseDict["RightForearm"]["tz"])
    return poseSeries


def colorPoint(idx):
    colorList = ['g', 'g', 'g', 'g', 'k', 'r', 'r', 'r', 'r', 'k']
    return colorList[idx]

def colorLine(idx):
    colorList = ['g', 'g', 'g', 'k', 'k', 'r', 'r', 'r', 'k', 'k']
    return colorList[idx]

def animate(pose_series, boxPosition=None, sequencesPara=None, pose_series_labels=None, reference_pose=None, fixed_poses=None, fixed_labels=["start", "end"], axis_limits=None, name=None):
    global ax, points, lines, fig, moving_labels, static_labels, reward, ani, factor, pointsDifference, sequences
    sequences = sequencesPara
    fig = pylab.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    w_in_inches = 4
    h_in_inches = 3
    fig.set_size_inches(w_in_inches, h_in_inches, True)


    # Reference pose
    #if reference_pose is not None:
    #    reference_cs = common.coordinate_system_for_affine(reference_pose.to_affine())
    #    reference_quiver = ax.quiver(*reference_cs.transpose(), colors=common.base_cs_colors, arrow_length_ratio=0.0)

    # Fixed poses
    static_labels = []
    #if fixed_poses is not None:
    #    fixed_quivers = []
    #    for idx, fixed_pose in enumerate(fixed_poses):
    #        pose_cs = common.coordinate_system_for_affine(fixed_pose.to_affine())
    #        fixed_quivers.append(ax.quiver(*pose_cs.transpose(), colors=common.fixed_cs_colors, length=0.05, arrow_length_ratio=0.0))
    #        x, y, _ = proj3d.proj_transform(fixed_pose.position.x, fixed_pose.position.y, fixed_pose.position.z, ax.get_proj())
    #        label = pylab.annotate(
    #            str(fixed_labels[idx]),
    #            xy=(x, y), xytext=(-10, 10),
    #            textcoords='offset points', ha='right', va='bottom',
    #            bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.5),
    #            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    #        static_labels.append(label)

    # Moving poses
    #pose_css = [[pose for pose in poses] for poses in pose_series] #common.coordinate_system_for_affine(pose.to_affine())
    if not (np.any(boxPosition == None)):
        phi = 45/180*3.14159
        boxOffsetX = -50
        boxOffsetY = -350
        boxX = boxPosition[0]*1000 * np.cos(phi) - np.sin(phi) * 1000*boxPosition[1] + boxOffsetX
        boxY = boxPosition[0]*1000 * np.sin(phi) + np.cos(phi) * 1000*boxPosition[1] + boxOffsetY
        boxZ = boxPosition[2]*1000+1200
        #boxMarker = ax.scatter(boxX, boxY, boxZ, marker='o', color='r', s=200)
        #boxMarkerN = ax.scatter(boxOffsetX, boxOffsetY, boxZ, marker='o', color='k', s=200)

    points = [ax.scatter(pose_series[i][0][0], pose_series[i][0][1], pose_series[i][0][2], marker='o', color=colorPoint(i)) for i in range(len(pose_series))]
    #points.append(ax.scatter((pose_series[3][0][0]+pose_series[5][0][0])/2, (pose_series[3][0][1]+pose_series[5][0][1])/2, (pose_series[3][0][2]+pose_series[5][0][2])/2, marker='o', color=colorPoint(9)))
    pointsDifference = np.zeros(len(pose_series))
    pointsDiff = [0 for i in range(len(pose_series) + 1)]
    #pointsDiff[0] = ax.scatter(0, 0, 0, marker='x')
    #pointsDiff[1] = ax.scatter(0, 0, 0, marker='x')
    #pointsDiff[7] = ax.scatter(0, 0, 0, marker='x')
    #pointsDiff[8] = ax.scatter(0, 0, 0, marker='x')
    #pointsDiff[9] = ax.scatter(0, 0, 0, marker='o', color='k', s=150)
    lines = [ax.plot([pose_series[i][0][0], pose_series[i + 1][0][0]], [pose_series[i][0][1], pose_series[i + 1][0][1]], zs=[pose_series[i][0][2], pose_series[i + 1][0][2]], color=colorLine(i)) for i in range(8)]
    lines.append(ax.plot([pose_series[3][0][0], pose_series[5][0][0]], [pose_series[3][0][1], pose_series[5][0][1]], zs=[pose_series[3][0][2], pose_series[5][0][2]], color=colorLine(8)))
    lines.append(ax.plot([pose_series[4][0][0], pose_series[9][0][0]], [pose_series[4][0][1], pose_series[9][0][1]], zs=[pose_series[4][0][2], pose_series[9][0][2]], color=colorLine(9)))
    
    #quivers = [ax.quiver(*pose_css[i][0].transpose(), colors=common.cs_colors, length=0.05, arrow_length_ratio=0.0) for i in range(len(pose_css))]

    # Labels
    if pose_series_labels is None:
        pose_series_labels = [str(i + 1) for i in range(len(pose_series))]
        pose_series_labels = ['lHand', 'lElbow', '', 'lShoulder', 'torso', 'rShoulder', '', 'rElbow', 'rHand', '']
        position = [(40, 10), (20, 30), (-0, 0), (-0, 50), (-0, -30), (0, 50), (-0, 0), (-20, 30), (-40, 10), (-0, 0),]
    moving_labels = []
    for idx, pose in enumerate([poses[0] for poses in pose_series]):
        if idx == 2 or idx == 6 or idx == 9:
            moving_labels.append(None)
            continue
        x, y, _ = proj3d.proj_transform(pose[0], pose[1], pose[2], ax.get_proj())
        label = pylab.annotate(
            pose_series_labels[idx],
            xy=(x, y), xytext=position[idx],
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        moving_labels.append(label)

    if axis_limits is None:
        axis_limits = default_axis_limits
    ax.set_xlim(axis_limits["x"][0], axis_limits["x"][1])
    ax.set_ylim(axis_limits["y"][0], axis_limits["y"][1])
    ax.set_zlim(axis_limits["z"][0], axis_limits["z"][1])
    ax.set_proj_type("persp")

    reward = 0

    def gen():
        global reward
        i = 0
        while reward <= 10:
            i += 1
            yield i

    def update(num):
        global points, lines, fig, moving_labels, static_labels, reward, factor, pointsDifference, sequences
        if num%(96/factor) == 0:
            print("num: ", num)
        num *= factor
        num += 0
        if num >= len(pose_series[0]):
            reward = 20
            plt.close(fig)
            print("also wrong if....")
            return
        if not(sequences is None) and sequences[num] == 0:
            print("in the wrong if...")
            return

        if fixed_poses is not None:
            for idx in range(len(static_labels)):
                pose = fixed_poses[idx]
                #pose_cs = common.coordinate_system_for_affine(pose.to_affine())
                #x, y, _ = proj3d.proj_transform(pose_cs[0, 0], pose_cs[0, 1], pose_cs[0, 2], ax.get_proj())
                #static_labels[idx].xy = x, y
                #static_labels[idx].update_positions(fig.canvas.renderer) 
        for series_idx in range(len(points)):
            step_idx = num % len(pose_series[0])
            new_cs = pose_series[series_idx][step_idx]
            points[series_idx].remove()
            points[series_idx] = ax.scatter(new_cs[0], new_cs[1], new_cs[2], marker='o', color=colorPoint(series_idx))
            if (series_idx == 0 or series_idx == 1 or series_idx == 7 or series_idx == 8) and False:
                pointsDiff[series_idx].remove()
                old_cs = pose_series[series_idx][step_idx - factor]
                pointsDifference[series_idx] = ((new_cs[0] - old_cs[0])**2 + (new_cs[1] - old_cs[1])**2 + (new_cs[2] - old_cs[2])**2)**0.5
                pointsDiff[series_idx] = ax.scatter((new_cs[0] - old_cs[0]) * 10 + series_idx * 0, (new_cs[1] - old_cs[1]) * 10, (new_cs[2] - old_cs[2]) * 10, marker='x', color=colorPoint(series_idx))

            x, y, _ = proj3d.proj_transform(new_cs[0], new_cs[1], new_cs[2], ax.get_proj())
            if series_idx == 2 or series_idx == 6 or series_idx == 9:
                continue
            moving_labels[series_idx].xy = x, y
            #moving_labels[series_idx].update_positions(fig.canvas.renderer)
        counter = 4
        maxDiff = 0
        pointsDifferenceMean = 0
        for i in [0, 1, 7, 8]:
            if np.isnan(pointsDifference[i]):
                counter -= 1
            else:
                pointsDifferenceMean += pointsDifference[i]
                if pointsDifference[i] > maxDiff:
                    maxDiff = pointsDifference[i]
        if counter > 1:
            pointsDifferenceMean = (pointsDifferenceMean - maxDiff)/(counter-1)
        elif counter > 0:
            pass # do nothing
        else:
            pass # Error case
        if pointsDifferenceMean < 10:
            colorHalt = 'g'
        else:
            colorHalt = 'r'

        #pointsDiff[9].remove()
        #pointsDiff[9] = ax.scatter(0, 0, 0, marker='o', color=colorHalt, s=150)
        for series_idx in range(8):
            step_idx = num % len(pose_series[0])
            new_cs_start = pose_series[series_idx][step_idx]
            new_cs_end = pose_series[series_idx + 1][step_idx]
            lines[series_idx][0].remove()
            lines[series_idx] = ax.plot([new_cs_start[0], new_cs_end[0]], [new_cs_start[1], new_cs_end[1]], zs=[new_cs_start[2], new_cs_end[2]], color=colorLine(series_idx))
        lines[8][0].remove()
        lines[9][0].remove()
        lines[8] = ax.plot([pose_series[3][step_idx][0], pose_series[5][step_idx][0]], [pose_series[3][step_idx][1], pose_series[5][step_idx][1]], zs=[pose_series[3][step_idx][2], pose_series[5][step_idx][2]], color=colorLine(8))
        lines[9] = ax.plot([pose_series[4][step_idx][0], pose_series[9][step_idx][0]], [pose_series[4][step_idx][1], pose_series[9][step_idx][1]], zs=[pose_series[4][step_idx][2], pose_series[9][step_idx][2]], color=colorLine(9))
        fig.canvas.draw()
        ax.view_init(elev=30., azim=240) #azim um z, links händisch

    # Set up formatting for the movie files
    factor = 6
    sampleTime = np.round(1000/96 * factor)

    ifSave = False
    if ifSave:
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        print("len(pose_series[0])/factor-1: ", int(len(pose_series[0])/factor-1))
        line_ani = animation.FuncAnimation(fig, update, interval=5, frames=int(len(pose_series[0])/factor-1))
        line_ani.save('test_'+ str(name) +'.mp4', writer=writer, dpi=320)
    else:
        ani = FuncAnimation(fig, update, frames=gen, interval=5, repeat=False)
        plt.show() 

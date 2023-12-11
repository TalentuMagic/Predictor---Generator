import csv
import random
import concurrent.futures
import time
import math
from tqdm import tqdm
import pandas as pd

line1 = list()
line2 = list()
line3 = list()
line4 = list()
line5 = list()
line6 = list()
line7 = list()
line8 = list()


def getRandomInstance():
    global line1, line2, line3, line4, line5, line6, line7, line8
    lines = list()
    bomb = "*"
    crystal = "$"
    currentLine = 0
    noBombs = 4

    # line 1
    if currentLine == 0:
        lines.append("".join(random.sample(
            [bomb, crystal], counts=[noBombs, 5], k=9)))
        line1.append(lines[0])
        currentLine += 1
        noBombs -= 1
    if currentLine == 1:
        lines.append("".join(random.sample(
            ["1", "2", "3", "4", "5", "6", "7", "8", "9"], k=1)))
        currentLine += 1

    # line 2
    if currentLine == 2:
        lines.append("".join(random.sample(
            [bomb, crystal], counts=[noBombs, 5], k=8)))
        line2.append(lines[2])
        currentLine += 1
        noBombs -= 1
    if currentLine == 3:
        lines.append("".join(random.sample(
            ["1", "2", "3", "4", "5", "6", "7", "8"], k=1)))
        currentLine += 1

    # line 3
    if currentLine == 4:
        lines.append("".join(random.sample(
            [bomb, crystal], counts=[noBombs, 5], k=7)))
        line3.append(lines[4])
        currentLine += 1
    if currentLine == 5:
        lines.append("".join(random.sample(
            ["1", "2", "3", "4", "5", "6", "7"], k=1)))
        currentLine += 1
    # line 4
    if currentLine == 6:
        lines.append("".join(random.sample(
            [bomb, crystal], counts=[noBombs, 4], k=6)))
        line4.append(lines[6])
        currentLine += 1
    if currentLine == 7:
        lines.append("".join(random.sample(
            ["1", "2", "3", "4", "5", "6"], k=1)))
        currentLine += 1
    # line 5
    if currentLine == 8:
        lines.append("".join(random.sample(
            [bomb, crystal], counts=[noBombs, 3], k=5)))
        line5.append(lines[8])
        currentLine += 1
        noBombs -= 1
    if currentLine == 9:
        lines.append("".join(random.sample(
            ["1", "2", "3", "4", "5"], k=1)))
        currentLine += 1

    # line 6
    if currentLine == 10:
        lines.append("".join(random.sample(
            [bomb, crystal], counts=[noBombs, 3], k=4)))
        line6.append(lines[10])
        currentLine += 1
    if currentLine == 11:
        lines.append("".join(random.sample(
            ["1", "2", "3", "4"], k=1)))
        currentLine += 1
    if currentLine == 12:
        lines.append("".join(random.sample(
            [bomb, crystal], counts=[noBombs, 2], k=3)))
        line7.append(lines[12])
        currentLine += 1
    if currentLine == 13:
        lines.append("".join(random.sample(
            ["1", "2", "3"], k=1)))
        currentLine += 1
    if currentLine == 14:
        lines.append("".join(random.sample(
            [bomb, crystal], counts=[noBombs, 1], k=2)))
        line8.append(lines[14])
        currentLine += 1
    if currentLine == 15:
        lines.append("".join(random.sample(
            ["1", "2"], k=1)))
        currentLine += 1

    score = 0.0
    j = 1
    for i in range(0, len(lines), 2):
        if lines[i][int(lines[j])-1] == "$":
            score += 12.5
            j += 2
        else:
            break
        # if lines[i][int(lines[j-2])-1] == "$" and lines[i][int(lines[j])-1] == "*" and i > 0:
        #     break

    lines.append(math.floor(score))

    return lines


def linesAs(dataSet, line, count):
    for l, li in enumerate(line):
        if dataSet == li and len(line) == 126:
            dataSet.append(l + count)
        elif dataSet == li and len(line) == 56:
            dataSet.append(l + count + 126)


def main():
    global line1, line2, line3, line4, line5, line6, line7, line8

    startTime = time.time()
    aux = list()
    print("Working on...", end="")
    with open("dataSet.csv", 'w', newline='') as file:
        wr = csv.writer(file)
        header = ["Line1", "Choice1", "Line2", "Choice2", "Line3", "Choice3", "Line 4", "Choice4",
                  "Line5", "Choice5", "Line6", "Choice6", "Line7", "Choice7", "Line8", "Choice8", "WinningChance", "Line1ID", "Line2ID", "Line3ID", "Line4ID", "Line5ID", "Line6ID", "Line7ID", "Line8ID"]
        wr.writerow(header)
    for _ in tqdm(range(1)):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = [executor.submit(getRandomInstance)
                      for _ in range(48_600)]  # 48_600
        with open("dataSet.csv", 'a', newline='') as file:
            for output in concurrent.futures.as_completed(future):
                wr = csv.writer(file)
                if output.result():
                    wr.writerow(output.result())
    dataSet = list()
    with open('dataSet.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            dataSet.append(row)
    for i in range(1, len(dataSet), 1):
        if dataSet[i][0] not in line1:
            line1.append(dataSet[i][0])
        if dataSet[i][2] not in line2:
            line2.append(dataSet[i][2])
        if dataSet[i][4] not in line3:
            line3.append(dataSet[i][4])
        if dataSet[i][6] not in line4:
            line4.append(dataSet[i][6])
        if dataSet[i][8] not in line5:
            line5.append(dataSet[i][8])
        if dataSet[i][10] not in line6:
            line6.append(dataSet[i][10])
        if dataSet[i][12] not in line7:
            line7.append(dataSet[i][12])
        if dataSet[i][14] not in line8:
            line8.append(dataSet[i][14])
    count = 126
    for i in tqdm(range(1, len(dataSet), 1)):
        for l in range(len(line1)):
            if dataSet[i][0] == line1[l]:
                dataSet[i].append(l + count)
        for l in range(len(line2)):
            if dataSet[i][2] == line2[l]:
                dataSet[i].append(l + count + len(line1))
        for l in range(len(line3)):
            if dataSet[i][4] == line3[l]:
                dataSet[i].append(l + count + len(line1) + len(line2))
        for l in range(len(line4)):
            if dataSet[i][6] == line4[l]:
                dataSet[i].append(
                    l + count + len(line1) + len(line2) + len(line3))
        for l in range(len(line5)):
            if dataSet[i][8] == line5[l]:
                dataSet[i].append(
                    l + count + len(line1) + len(line2) + len(line3) + len(line4))
        for l in range(len(line6)):
            if dataSet[i][10] == line6[l]:
                dataSet[i].append(
                    l + count + len(line1) + len(line2) + len(line3) + len(line4) + len(line5))
        for l in range(len(line7)):
            if dataSet[i][12] == line7[l]:
                dataSet[i].append(
                    l + count + len(line1) + len(line2) + len(line3) + len(line4) + len(line5) + len(line6))
        for l in range(len(line8)):
            if dataSet[i][14] == line8[l]:
                dataSet[i].append(
                    l + count + len(line1) + len(line2) + len(line3) + len(line4) + len(line5) + len(line6) + len(line7))
    with open('dataSet.csv', 'w', newline='') as file:
        wr = csv.writer(file)
        for row in dataSet:
            wr.writerow(row)

    print("DONE")
    print(
        f"* It took {round((time.time()-startTime)/60,2)} min(s) to complete.")


if __name__ == "__main__":
    main()

# 15 Test Cases

These test cases cover different situations required for the assignment. The original PathFinder sample is used as **TC01**.

1. **TC01 – Original PathFinder Sample Graph**  
   Uses the provided sample graph file `PathFinder-test.txt` with origin 2 and destinations 5 and 4.

2. **TC02 – Single Destination Simple Path**  
   A small graph with only one destination and one clear valid path.

3. **TC03 – Multiple Destinations Available**  
   A graph where more than one goal node exists and the algorithm should stop at the first reachable goal.

4. **TC04 – Direct Edge to Goal**  
   The origin has a direct connection to the destination.

5. **TC05 – No Reachable Goal**  
   The destination exists but there is no valid path from the origin.

6. **TC06 – Start Node Is Destination**  
   The origin node is already one of the destination nodes.

7. **TC07 – Directed Edge Restriction**  
   Tests whether the program respects one-way edges and avoids illegal reverse movement.

8. **TC08 – Tie Breaking by Smaller Node Number**  
   Two or more nodes have equal evaluation values, so the smaller numbered node should be expanded first.

9. **TC09 – Tie Breaking by Chronological Insertion Order**  
   Nodes with equal values on different branches should be expanded according to insertion order.

10. **TC10 – Deep Search Case for DFS**  
    A graph designed to show DFS going deeper before backtracking.

11. **TC11 – Wide Search Case for BFS**  
    A graph designed to show BFS expanding level by level.

12. **TC12 – Cycle in Graph**  
    A graph containing loops to test that the search avoids infinite repetition.

13. **TC13 – Different Edge Costs**  
    A graph where the lowest-cost path is different from the path with fewer moves.

14. **TC14 – Equal Cost Alternative Paths**  
    Two or more valid paths have the same total evaluation, testing stable tie breaking.

15. **TC15 – Large Graph Performance Case**  
    A larger graph to test whether all algorithms still return valid output in reasonable time.

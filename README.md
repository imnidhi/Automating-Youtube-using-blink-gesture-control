### Automating Youtube using blink detection and gesture control
This project aims to automate a widely used software Youtube by using two basic features- blink detection and gesture control through a simple web camera. We try to understand how people use a well known software such as Youtube and thus automate it to reduce time involved in physical human interaction with the system and thus achieve the same results with fewer efforts.
#### Working
The user input is divided into two categories - a blink or a gesture . When an input is received in the form of a left blink or a right blink, this data is used to control the playback speed of the Youtube video playing. A single left blink will reduce the speed by 0.25x whereas a single right blink will increase the speed by 0.25x accordingly. If the user input is a hand gesture this will automatically either play/ pause/ mute/ unmute the video based on the respective hand gesture.
* A closed fist corresponds to a pause.
* Gesture showing the index finger corresponds to mute.
* Gesture showing all five fingers correspond to either playing a paused video or unmuting a muted video
 
At any point the user can choose to exit the detection process by simply pressing the ‘q’ key on the keyboard and the system can no longer detect a blink or a gesture. The script will need to be run again to enable the same.

#### Limitations
The limitations of this project is bound by noise in a video stream or incorrect facial landmarks or fast changes in viewing angle which could report a blink even though it hasn't occurred in reality. A small cavity within the region of interest might show an incorrect value of convexity defect and this might deviate from the expected result.  Thus we can use more sophisticated methods to implement the blink detection such as training a SVM or gesture control such as using a CNN classifier which will result in increased accuracy of detection of the user input.


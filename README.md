# CS5590-0001-Project

 One Circle created by Team 1.
 
## One Circle
### Team details

 ●    Jingtang Ma (Arthur): Senior in Computer Science & Minor in Math; University of Missouri -- Kansas City; Contact: jmkqf@umsystem.edu
 ●    Molan Zhang: Graduate in Computer Science; University of Missouri -- Kansas City; mz9kk@mail.umkc.edu
 ●    Qiao Yang: Senior in Information Techonogly; University of Missouri -- Kansas City; qjy2fc@mail.umkc.edu
 ●    Benjamin Nguyen: Computer Science Major; University of Missouri -- Kansas City, pdnrtv@umsystem.edu

 ### The story and its details
 TODO 
 
 ### The data and its details

 In this project I use the Labeled Faces in the Wild Dataset[1] as our training and testing dataset. This is a face photo database designed for studying the problem of face recognition in the natural environment. The dataset contains more than 13,000 images collected from the Internet. Each face is marked with the person’s name. This dataset contains a total 5749 person’s photos, of which 1,680 people have two or more different photos, which are captured by the Viola-Jones face detector.  Some of samples can be seen below:
![Image text](https://github.com/CS5590-0001-Projject/CS5590-0001-Project/blob/main/screenshot/dataset.JPG)

 Given that the ideal application scenario of this model is the company's commuting records and the pursuit of suspects by the public security organs, the input should come from real life. For the company, the input photos of the model may be taken in the office building. For public security organs, the pictures are more from roads, shopping malls and other public places. 

 ### Working screens from project
 
 ●    Training process
 ![Image text](https://github.com/CS5590-0001-Projject/CS5590-0001-Project/blob/main/screenshot/training_process.JPG)
 
 This image shows the training process. Here I trained the Triplet network for 10 epches. And for the linear classification network I trained 100 epches. It takes about one hour.
 
 ●    Training and validation loss rate
 ![Image text](https://github.com/CS5590-0001-Projject/CS5590-0001-Project/blob/main/screenshot/loss.JPG)
 
 ●    Training and validation accuracy rate
 ![Image text](https://github.com/CS5590-0001-Projject/CS5590-0001-Project/blob/main/screenshot/acc.JPG)
 
 ●    Test result analysis
        ●    Confusion matrix
        ●    Classification report
        ●    Conlusion
        
●    Some testing samples


### Work sharing/Module sharing between teammates
 TODO 
 
### Any issues, blockages with the project
 TODO 
 
### GitHub link for your project
 ●    https://github.com/jun0405/umkc_one_circle
 
 ### References
 ●    Huang, G. B., Mattar, M., Berg, T., & Learned-Miller, E. (2008, October). Labeled faces in the wild: A database for studying face recognition in unconstrained environments.


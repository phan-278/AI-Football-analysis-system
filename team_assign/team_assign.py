from sklearn.cluster import KMeans

class TeamAssign:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        
        # Reshape image into array
        img = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2,init='k-means++',n_init=1)
        kmeans.fit(img)

        return kmeans


    def get_player_color(self, frame,bbox):
        # Get the player image
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        # Take the shirt partern from player
        shirt = image[0: int (image.shape[0]/2),:]
        
        # Get cluster model (KMeans)
        kmeans = self.get_clustering_model(shirt)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to image shape
        clutered_img = labels.reshape(shirt.shape[0],shirt.shape[1])

        # Get the player cluster
        corner_clusters = [clutered_img[0,0],clutered_img[-1,0],clutered_img[0,-1],clutered_img[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key = corner_clusters.count)
        player_cluster = 1- non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
    
    def assign_team_color(self,frame,player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame,bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters = 2,init = 'k-means++',n_init = 10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def assign_player_into_team(self, frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id
    


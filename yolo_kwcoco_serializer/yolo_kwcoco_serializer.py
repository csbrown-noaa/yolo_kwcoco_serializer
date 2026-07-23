import os
import kwcoco
import numpy as np

# A hardcoded tuple of video formats natively supported by Ultralytics.
# This prevents needing to import the heavy Ultralytics engine just for format checking.
VID_FORMATS = (
    'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'
)


class Yolo2KwcocoSerializer:
    """
    A stateful serializer that converts Ultralytics YOLO Results objects 
    into a strictly compliant KWCOCO dataset manifest.

    This class handles the automatic routing of images versus video frames 
    based on file extensions, maps bounding boxes to the COCO standard, 
    captures tracker IDs, and optionally captures rich probability distributions.

    Parameters
    ----------
    categories : dict
        A dictionary mapping class IDs to class names (e.g., model.names).

    Example
    -------
    >>> import numpy as np
    >>> serializer = Yolo2KwcocoSerializer(categories={0: 'cat', 1: 'dog'})
    >>> print(sorted(list(serializer.dset.name_to_cat.keys())))
    ['cat', 'dog']
    
    >>> # Create a lightweight mock of Ultralytics Results to test without PyTorch
    >>> class MockTensor:
    ...     def __init__(self, val): self.val = val
    ...     def cpu(self): return self
    ...     def numpy(self): return np.array(self.val)
    >>> class MockBoxes:
    ...     def __init__(self):
    ...         self.xyxy = MockTensor([[10, 10, 30, 40]]) # x1, y1, x2, y2
    ...         self.conf = MockTensor([0.95])
    ...         self.cls = MockTensor([1])
    ...         self.id = MockTensor([5]) # Adding mock track ID
    ...     def __len__(self): return 1
    >>> class MockResult:
    ...     path = "/fake/path/test_image.jpg"
    ...     orig_shape = (480, 640)
    ...     boxes = MockBoxes()
    
    >>> # Process the mock result
    >>> serializer.add_result(MockResult())
    
    >>> # Verify the KWCOCO mapping (bbox should be [x, y, w, h] format)
    >>> ann = list(serializer.dset.anns.values())[0]
    >>> ann['category_id']
    1
    >>> ann['bbox']
    [10.0, 10.0, 20.0, 30.0]
    >>> ann['track_id']
    5
    >>> img = list(serializer.dset.imgs.values())[0]
    >>> img['file_name']
    'test_image.jpg'
    """

    def __init__(self, categories: dict):
        self.dset = kwcoco.CocoDataset()
        
        # State trackers for video processing
        self.current_video_path = None
        self.current_video_id = None
        self.frame_counter = 0

        # State tracker for unique track IDs to prevent un-registered track warnings
        self.registered_tracks = set()

        self._initialize_categories(categories)

    def _initialize_categories(self, categories: dict):
        """
        Populates the KWCOCO dataset with the predefined categories.

        Parameters
        ----------
        categories : dict
            A dictionary mapping class IDs (int) to class names (str).
        """
        for cls_id, cls_name in categories.items():
            self.dset.add_category(name=cls_name, id=cls_id)

    def _is_video(self, path: str) -> bool:
        """
        Determines if a given file path is a video based on its extension.

        Parameters
        ----------
        path : str
            The file path or URL to check.

        Returns
        -------
        bool
            True if the file is a supported video format, False otherwise.

        Example
        -------
        >>> serializer = Yolo2KwcocoSerializer({0: 'cat'})
        >>> serializer._is_video("camera1.mp4")
        True
        >>> serializer._is_video("photo.JPG")
        False
        >>> serializer._is_video("stream.avi")
        True
        """
        ext = path.split('.')[-1].lower()
        return ext in VID_FORMATS

    def _register_new_video(self, source_path: str):
        """
        Registers a new video entity in the KWCOCO dataset and updates internal state.

        Parameters
        ----------
        source_path : str
            The raw source path of the video.
        """
        video_name = os.path.basename(source_path)
        
        # Add video to dataset (uses leaf name as ID per our architecture)
        self.current_video_id = self.dset.add_video(name=video_name)
        
        self.current_video_path = source_path
        self.frame_counter = 0

    def _process_image_level(self, result) -> int:
        """
        Registers the image or video frame in the dataset.

        Parameters
        ----------
        result : ultralytics.engine.results.Results
            The Ultralytics results object for a single frame or image.

        Returns
        -------
        int
            The internal KWCOCO image_id for the newly registered frame.
        """
        path = result.path
        is_vid = self._is_video(path)
        
        height, width = result.orig_shape

        if is_vid:
            # Check if we transitioned to a new video stream
            if path != self.current_video_path:
                self._register_new_video(path)
            
            # Register a video frame with a generated name
            video_name = os.path.basename(path)
            image_id = self.dset.add_image(
                name=f"{video_name}_frame_{self.frame_counter:07d}",
                video_id=self.current_video_id,
                frame_index=self.frame_counter,
                width=width,
                height=height
            )
            self.frame_counter += 1
            
        else:
            # Register a standalone image (requires file_name to be the leaf name)
            leaf_name = os.path.basename(path)
            image_id = self.dset.add_image(
                file_name=leaf_name,
                name=leaf_name,
                width=width,
                height=height
            )
            
        return image_id

    def add_result(self, result):
        """
        Consumes an Ultralytics Results object, extracts all spatial, semantic, 
        and tracking data, and appends it to the KWCOCO manifest.

        Parameters
        ----------
        result : ultralytics.engine.results.Results
            The output object from model.predict() or model.track().
        """
        # 1. Register the image/frame
        image_id = self._process_image_level(result)
        
        # 2. Dispatching / Data Modality Checks
        if getattr(result, 'masks', None) is not None:
            raise NotImplementedError("Segmentation mask serialization is planned but not yet implemented.")
        if getattr(result, 'keypoints', None) is not None:
            raise NotImplementedError("Pose/Keypoint serialization is planned but not yet implemented.")
        if getattr(result, 'probs', None) is not None:
            raise NotImplementedError("Whole-image classification serialization is planned but not yet implemented.")
            
        # Check if any bounding box detections exist in this frame
        if getattr(result, 'boxes', None) is None or len(result.boxes) == 0:
            return

        # 3. Extract spatial data from GPU/CPU to native python/numpy types
        # Ultralytics xyxy format: [x_min, y_min, x_max, y_max]
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Extract Tracker IDs (if model.track() was used)
        track_ids = None
        if getattr(result.boxes, 'id', None) is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        # Check for custom probability arrays
        soft_scores = None
        if hasattr(result, 'soft_scores'):
            soft_scores = result.soft_scores.cpu().numpy()

        # 4. Add each detection as an annotation
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            
            # Convert to COCO standard: [top_left_x, top_left_y, width, height]
            w = x2 - x1
            h = y2 - y1
            coco_bbox = [float(x1), float(y1), float(w), float(h)]
            
            cat_id = int(classes[i])
            score = float(confs[i])
            
            # Build the annotation kwargs
            ann_kwargs = {
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': coco_bbox,
                'score': score
            }
            
            # Inject tracker ID if present
            if track_ids is not None:
                track_id = int(track_ids[i])
                
                # Formally register the track ID with kwcoco to satisfy schema constraints
                if track_id not in self.registered_tracks:
                    self.dset.add_track(name=str(track_id), id=track_id)
                    self.registered_tracks.add(track_id)
                    
                ann_kwargs['track_id'] = track_id
            
            # Inject standard KWCOCO 'prob' field if we have soft scores
            if soft_scores is not None:
                ann_kwargs['prob'] = soft_scores[i].tolist()
                
            self.dset.add_annotation(**ann_kwargs)

    def save(self, filepath: str):
        """
        Writes the completed KWCOCO manifest to disk.

        Parameters
        ----------
        filepath : str
            The destination path for the output JSON file.
        """
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Dump state to JSON
        self.dset.dump(filepath, newlines=True)
        print(f"✅ KWCOCO dataset successfully saved to {filepath}")

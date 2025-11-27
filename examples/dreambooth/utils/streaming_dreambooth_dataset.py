class StreamingDreamBoothDataset(Dataset):
    """
    A streaming dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It streams from a HuggingFace dataset and processes images on-the-fly.
    """

    def __init__(
        self,
        dataset_urls,  # List of URLs for webdataset tar files
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        cache_dir=None,
        dataset_length=None,  # Approximate length for __len__
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. "
                "Please install the datasets library: `pip install datasets`."
            )
        
        self.size = size
        self.center_crop = center_crop
        self.repeats = repeats
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.dataset_length = dataset_length or 100000  # Default approximate length
        
        # Load streaming dataset
        self.streaming_dataset = load_dataset(
            "webdataset", 
            data_files={"train": dataset_urls}, 
            split="train", 
            streaming=True,
            cache_dir=cache_dir,
        )
        
        # Create an iterator that we can cycle through
        self._dataset_iterator = iter(self.streaming_dataset)
        self._current_items = []  # Cache for current batch of items
        self._cache_size = 1000  # Number of items to cache
        self._current_index = 0
        
        # Pre-load initial batch
        self._refill_cache()
        
        # Setup class images if provided
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
        else:
            self.class_data_root = None
            self.num_class_images = 0

        # Image transforms (no data augmentation as requested)
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _refill_cache(self):
        """Refill the cache with new items from the streaming dataset."""
        new_items = []
        try:
            for _ in range(self._cache_size):
                item = next(self._dataset_iterator)
                # Repeat each item according to repeats parameter
                for _ in range(self.repeats):
                    new_items.append(item)
        except StopIteration:
            # If we reach the end, restart the iterator
            self._dataset_iterator = iter(self.streaming_dataset)
            # Try to get at least some items
            try:
                for _ in range(min(self._cache_size, 100)):
                    item = next(self._dataset_iterator)
                    for _ in range(self.repeats):
                        new_items.append(item)
            except StopIteration:
                # If still no items, something is wrong
                if not new_items and not self._current_items:
                    raise RuntimeError("Unable to load any items from the streaming dataset")
        
        self._current_items.extend(new_items)
        
    def __len__(self):
        # For streaming datasets, we need to provide an approximate length
        # This can be adjusted based on your needs
        return self.dataset_length * self.repeats

    def __getitem__(self, index):
        # Check if we need to refill cache
        cache_index = index % len(self._current_items) if self._current_items else 0
        
        # If we're running low on cached items, refill
        if cache_index >= len(self._current_items) - 100:
            self._refill_cache()
            cache_index = index % len(self._current_items)
        
        # Get the streaming item
        stream_item = self._current_items[cache_index]
        
        example = {}
        
        # Process instance image
        instance_image = stream_item['jpg']  # PIL Image
        instance_image = exif_transpose(instance_image)
        
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        # Apply transforms
        instance_image = self.image_transforms(instance_image)
        example["instance_images"] = instance_image
        
        # Get prompt from the dataset or use default
        if 'json' in stream_item and 'prompt' in stream_item['json']:
            example["instance_prompt"] = stream_item['json']['prompt']
        else:
            example["instance_prompt"] = self.instance_prompt
        
        # Handle class images if provided
        if self.class_data_root and self.num_class_images > 0:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)
            
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt
        
        return example


# Convenience function to create the streaming dataset with the same URLs as in your example
def create_streaming_dreambooth_dataset(
    instance_prompt,
    class_prompt,
    class_data_root=None,
    class_num=None,
    size=1024,
    repeats=1,
    center_crop=False,
    cache_dir="/pscratch/sd/g/gabeguo/cache/huggingface",
    dataset_length=100000,
):
    """
    Create a StreamingDreamBoothDataset using the text-to-image-2M dataset.
    """
    base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
    num_shards = 46  # Number of webdataset tar files
    urls = [base_url.format(i=i) for i in range(num_shards)]
    
    return StreamingDreamBoothDataset(
        dataset_urls=urls,
        instance_prompt=instance_prompt,
        class_prompt=class_prompt,
        class_data_root=class_data_root,
        class_num=class_num,
        size=size,
        repeats=repeats,
        center_crop=center_crop,
        cache_dir=cache_dir,
        dataset_length=dataset_length,
    )
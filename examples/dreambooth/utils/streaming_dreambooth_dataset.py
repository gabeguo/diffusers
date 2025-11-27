class StreamingDreamBoothDataset(Dataset):
    """
    A streaming dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It streams from a HuggingFace dataset and processes images on-the-fly.
    """

    def __init__(
        self,
        dataset_urls,  # List of URLs for webdataset tar files
        instance_prompt=None,
        class_prompt=None,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        cache_dir=None,
        dataset_length=2200000,
        cache_size=128,
        rank=0,
        world_size=1,
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
        self.dataset_length = dataset_length

        self.rank = rank
        self.world_size = world_size 

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
        self._cache_size = cache_size  # Number of items to cache
        self._current_cache_index = 0  # Track position within current cache
        
        self._dataset_iterator.skip(self.rank) # stagger the starting point of each GPU, so they don't overlap
        
        # Pre-load initial batch
        self._refill_cache()

        assert instance_prompt is None, "Instance prompt not needed"
        assert class_prompt is None, "Class prompt is not supported for streaming dataset"
        assert class_data_root is None, "Class data root is not supported for streaming dataset"
        assert class_num is None, "Class number is not supported for streaming dataset"
        assert repeats == 1, "Repeats is not supported for streaming dataset"

        # Image transforms (no data augmentation as requested)
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _refill_cache(self):
        """Refill the cache with new items from the streaming dataset."""
        # Clear the old cache to free memory
        self._current_items = list()
        # Reset cache index
        self._current_cache_index = 0
        
        def fill_cache():
            for _ in range(self._cache_size):
                for _ in range(self.world_size):
                    item = next(self._dataset_iterator) # each GPU sees different data
                # Repeat each item according to repeats parameter
                for _ in range(self.repeats):
                    self._current_items.append(item)
        try:
            fill_cache()
        except StopIteration:
            # If we reach the end, restart the iterator
            self._dataset_iterator = iter(self.streaming_dataset)
            self._dataset_iterator.skip(self.rank) # stagger the starting point of each GPU
            fill_cache()
        
    def __len__(self):
        # For streaming datasets, we need to provide an approximate length
        # This can be adjusted based on your needs
        return self.dataset_length * self.repeats

    def __getitem__(self, index):
        # If we've gone through all items in the current cache, refill it
        if self._current_cache_index >= len(self._current_items):
            self._refill_cache()
            assert self._current_cache_index == 0, "Cache index should be 0 after refill"
        
        # Get the streaming item
        stream_item = self._current_items[self._current_cache_index]
        
        # Increment cache index for next access
        self._current_cache_index += 1
        
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
            raise ValueError("Prompt not found in the dataset")
        
        return example


# Convenience function to create the streaming dataset with the same URLs as in your example
def create_streaming_dreambooth_dataset(
    instance_prompt,
    class_prompt=None,
    class_data_root=None,
    class_num=None,
    size=1024,
    repeats=1,
    center_crop=False,
    cache_dir="/pscratch/sd/g/gabeguo/cache/huggingface",
    dataset_length=100000,
    rank=0,
    world_size=1,
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
        rank=rank,
        world_size=world_size,
    )
"""Upload forms for the dashboard."""

from django import forms


class MultiFileInput(forms.FileInput):
    """FileInput that supports selecting multiple files.

    Django 6 removed 'multiple' from built-in FileInput/ClearableFileInput.
    This subclass re-enables it via allow_multiple_selected.
    """
    allow_multiple_selected = True


class ImageUploadForm(forms.Form):
    """Form for single image upload."""
    image = forms.ImageField(
        label="Upload Steel Surface Image",
        help_text="Supported formats: JPG, PNG. Max 10 MB.",
        widget=forms.ClearableFileInput(attrs={
            "class": "form-control",
            "accept": "image/*",
            "id": "image-upload-input",
        }),
    )


class BatchUploadForm(forms.Form):
    """Form for batch image upload (multiple files)."""
    images = forms.FileField(
        label="Upload Batch Images",
        help_text="Select multiple steel surface images for batch inspection.",
        widget=MultiFileInput(attrs={
            "class": "form-control",
            "accept": "image/*",
            "id": "batch-upload-input",
        }),
    )
    batch_name = forms.CharField(
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "Enter batch name (optional)",
            "id": "batch-name-input",
        }),
    )

    def clean_images(self):
        """Accept multiple files from the multi-file input."""
        files = self.files.getlist("images")
        if not files:
            raise forms.ValidationError("Please select at least one image file.")
        return files

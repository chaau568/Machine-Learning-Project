from django.db import models #type: ignore

class User_Details(models.Model):
  model = models.CharField(max_length=20)
  predict = models.CharField(max_length=20)
  confidence = models.CharField(max_length=20)
  created_at = models.DateTimeField(auto_now_add=True)

  def __str__(self):
      return self.model + " predict: " + self.predict + " confidence: " + self.confidence


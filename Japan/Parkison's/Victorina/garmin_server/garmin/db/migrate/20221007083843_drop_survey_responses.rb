class DropSurveyResponses < ActiveRecord::Migration[5.2]
  def up
    drop_table :survey_responses
  end

  def down
    create_table :survey_responses do |t|
      t.references :user, foreign_key: true
      t.jsonb :answers, null: false, default: {}

      t.timestamps
    end
  end
end
